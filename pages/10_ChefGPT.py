import secrets
from typing import List, Dict
import logging
from datetime import timedelta, datetime
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from pydantic import BaseModel
import requests
from jwtforchef import create_access_token, decode_token_to_get_username
from dotenv import load_dotenv

# Initialize Pinecone and OpenAI embeddings
load_dotenv()
pc = Pinecone(api_key="PINECONE_API_KEY")
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_existing_index(
    "recipes",
    embeddings,
)

# In-memory "database"
users_db: Dict[str, Dict] = {}
token_db: Dict[str, str] = {}
access_tokens_db: Dict[str, Dict] = {}

ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="ChefGPT. The best provider of Indian Recipes in the world",
    description=(
        "Give ChefGPT a couple of ingredients and it will give you recipes in"
        " return."
    ),
    servers=[
        {  # Make sure you don't have a slash at the end of the url
            "url": "https://apple-which-closest-leonard.trycloudflare.com",
        }
    ],
)
# Password hashing setup
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Document(BaseModel):
    page_content: str


class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: str


class FavoriteResponse(BaseModel):
    message: str


@app.get(
    "/recipes",
    summary="Return a list of recipes",
    description=(
        "Upon receiving one or multiple ingredients, this endpoint will return"
        " a list of recipes that contain the ingredients"
    ),
    response_description=(
        "A Document object that contains the recipe and preparation"
        " instructions"
    ),
    response_model=List[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Log In or Register</title>
    </head>
    <body>
        <h1>Log In or Register</h1>
        <form action="/authorize" method="post">
            <input type="hidden" name="grant_type" value="password">
            <input type="hidden" name="redirect_uri" value="{redirect_uri}">
            <input type="hidden" name="state" value="{state}">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" required><br><br>
            <label for="email">Email (only for registration):</label>
            <input type="email" id="email" name="email"><br><br>
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required><br><br>
            <button type="submit" name="action" value="login">Log In</button>
            <button type="submit" name="action" value="register">Register</button>
        </form>
    </body>
    </html>
    """


@app.post("/authorize", response_class=HTMLResponse, include_in_schema=False)
async def authorize_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(None),
    action: str = Form(...),
    redirect_uri: str = Form(...),
    state: str = Form(...),
):
    if action == "register":
        if email is None:
            return HTMLResponse(
                "Email is required for registration", status_code=400
            )
        if username in users_db:
            return HTMLResponse("Username already registered", status_code=400)

        hashed_password = pwd_context.hash(password)
        users_db[username] = {
            "username": username,
            "email": email,
            "password": hashed_password,
        }
        # Automatically log in the user after registration
        code = secrets.token_urlsafe(16)
        token_db[code] = username
        return RedirectResponse(
            url=f"{redirect_uri}?code={code}&state={state}", status_code=303
        )
    elif action == "login":
        user = users_db.get(username)
        if not user or not pwd_context.verify(password, user["password"]):
            return HTMLResponse(
                "Incorrect username or password", status_code=401
            )
        # Generate authorization code
        code = secrets.token_urlsafe(16)
        token_db[code] = username
        return RedirectResponse(
            url=f"{redirect_uri}?code={code}&state={state}", status_code=303
        )


@app.post("/token")
async def login_for_access_token(code: str = Form(...)):

    # Check if the code is valid
    username = token_db.get(code)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid code")
    # Check if an access token already exists and is still valid
    token_info = access_tokens_db.get(code)
    if token_info:
        access_token = token_info.get("access_token")
        expires = token_info.get("expires")
        if access_token and expires > datetime.utcnow():
            # Return the existing access token if it is still valid
            return {"access_token": access_token, "token_type": "bearer"}

    # Create the access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires = datetime.utcnow() + access_token_expires
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    # Store the new access token and its expiration time
    access_tokens_db[code] = {"access_token": access_token, "expires": expires}
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/add_favorite", response_model=FavoriteResponse)
async def add_favorite_recipe(request: Request, recipe: Recipe):
    token = request.headers.get("authorization", "").split()[1]
    if not token:
        logging.error("Authorization header is missing")
        raise HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    username = decode_token_to_get_username(token)
    if not username:
        logging.error("Invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid token")

    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    if "favorite_recipes" not in users_db[username]:
        users_db[username]["favorite_recipes"] = []
    users_db[username]["favorite_recipes"].append(recipe.dict())
    return {
        "status": "success",
        "message": "Recipe added to favorites",
        "recipe": recipe.dict(),
    }


@app.get(
    "/favorite",
    summary="Get favorite recipes",
    description="Retrieves the user's favorite recipes.",
    response_description="The user's favorite recipes",
    response_model=List[Recipe],
)
async def display_favorite_recipes(request: Request):
    token = request.headers.get("authorization", "").split()[1]
    if not token:
        logging.error("Authorization header is missing")
        raise HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    username = decode_token_to_get_username(token)
    if not username:
        logging.error("Invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid token")

    if (
        username not in users_db
        or "favorite_recipes" not in users_db[username]
    ):
        return HTMLResponse(
            "<html><body><h1>No favorite recipes found.</h1></body></html>"
        )

    favorite_recipes = users_db[username]["favorite_recipes"]

    html_content = "<html><head><title>Favorite Recipes</title></head><body>"
    html_content += "<h1>Favorite Recipes</h1>"
    if favorite_recipes:
        for recipe in favorite_recipes:
            html_content += f"<h2>{recipe['name']}</h2>"
            html_content += "<h3>Ingredients:</h3><ul>"
            for ingredient in recipe["ingredients"]:
                html_content += f"<li>{ingredient}</li>"
            html_content += "</ul>"
            html_content += (
                f"<h3>Instructions:</h3><p>{recipe['instructions']}</p>"
            )
    else:
        html_content += "<p>No favorite recipes found.</p>"
    html_content += "</body></html>"

    return HTMLResponse(content=html_content)
