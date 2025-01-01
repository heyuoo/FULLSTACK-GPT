from typing import List
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
import os, json, secrets, jwt
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from passlib.context import CryptContext

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
db_pathname = f"./.cache/chefgpt/"
if not os.path.exists(db_pathname):
    os.mkdir(db_pathname)
users_pathname = f"{db_pathname}users.json"
tokens_pathname = f"{db_pathname}tokens.json"
jwts_pathname = f"{db_pathname}jwts.json"
favorites_pathname = f"{db_pathname}favorites.json"
if not os.path.exists(users_pathname):
    file = open(users_pathname, "w")
    file.close()
if not os.path.exists(tokens_pathname):
    file = open(tokens_pathname, "w")
    file.close()
if not os.path.exists(jwts_pathname):
    file = open(jwts_pathname, "w")
    file.close()
if not os.path.exists(favorites_pathname):
    file = open(favorites_pathname, "w")
    file.close()
with open(users_pathname, "r") as f:
    try:
        users_db = json.load(f)
    except json.decoder.JSONDecodeError as jde:
        users_db = {}
with open(tokens_pathname, "r") as f:
    try:
        tokens_db = json.load(f)
    except json.decoder.JSONDecodeError as jde:
        tokens_db = {}
with open(jwts_pathname, "r") as f:
    try:
        jwts_db = json.load(f)
    except json.decoder.JSONDecodeError as jde:
        jwts_db = {}
with open(favorites_pathname, "r") as f:
    try:
        favorites_db = json.load(f)
    except json.decoder.JSONDecodeError as jde:
        favorites_db = {}
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_existing_index(
    "recipes",
    embeddings,
)
app = FastAPI(
    title="ChefGPT. The best provider of Indian Recipes in the world.",
    description=(
        "Give ChefGPT the name of an ingredient and it will give you multiple"
        " recipes to use that ingredient on in return."
    ),
    servers=[
        {
            "url": "https://culture-ordered-briefing-cost.trycloudflare.com",
        },
    ],
)
# Password hashing setup
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


class Document(BaseModel):
    page_content: str


class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: str


@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description=(
        "Upon receiving an ingredient, this endpoint will return a list of"
        " recipes that contain that ingredient."
    ),
    response_description=(
        "A Document object that contains the recipe and preparation"
        " instructions"
    ),
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


@app.get(
    "/favorites",
    summary="Returns a list of favorite recipes.",
    description="This endpoint will return a list of favorites of the user.",
)
def get_favorites(request: Request):
    token = request.headers.get("authorization", "").split()[1]
    payload = jwt.decode(token, key=PINECONE_API_KEY, algorithms=["HS256"])
    username = payload.get("username")
    favorites = favorites_db[username]
    return [favorite["name"] for favorite in favorites]


@app.post(
    "/favorites",
    summary="Sets recipe to favorite.",
    description="This endpoint will set the received recipe to favorites.",
)
def post_favorites(request: Request, recipe: Recipe):
    token = request.headers.get("authorization", "").split()[1]
    payload = jwt.decode(token, key=PINECONE_API_KEY, algorithms=["HS256"])
    username = payload.get("username")
    favorites = favorites_db[username]
    if recipe in favorites:
        return Response("This recipe is already in favorites.", 400)
    else:
        favorites.append(
            {
                "name": recipe.name,
                "ingredients": recipe.ingredients,
                "instructions": recipe.instructions,
            }
        )
        favorites_db[username] = favorites
        with open(favorites_pathname, "w") as f:
            json.dump(favorites_db, f)
        return Response("Recipe deleted.")


@app.delete(
    "/favorites",
    summary="Deletes recipe from favorite.",
    description=(
        "This endpoint will delete the received recipe from favorites."
    ),
)
def delete_favorites(request: Request, recipe_name: str):
    token = request.headers.get("authorization", "").split()[1]
    payload = jwt.decode(token, key=PINECONE_API_KEY, algorithms=["HS256"])
    username = payload.get("username")
    favorites = favorites_db[username]
    for favorite in favorites:
        if favorite["name"] == recipe_name:
            favorites.remove(favorite)
            favorites_db[username] = favorites
            with open(favorites_pathname, "w") as f:
                json.dump(favorites_db, f)
            return Response("Recipe deleted.")
    else:
        return Response("This recipe is not in favorites.", 404)


@app.get(
    "/auth",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Log In or Register</title>
        </head>
        <body>
            <h1>Log In or Register</h1>
            <form action="/auth" method="post">
                <input type="hidden" name="grant_type" value="password">
                <input type="hidden" name="redirect_uri" value="{redirect_uri}">
                <input type="hidden" name="state" value="{state}">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required><br><br>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required><br><br>
                <button type="submit" name="action" value="login">Log In</button>
                <button type="submit" name="action" value="register">Register</button>
            </form>
        </body>
    </html>
    """


def login(username, redirect_uri, state):
    # Generate authorization code
    code = secrets.token_urlsafe(16)
    tokens_db[code] = username
    with open(tokens_pathname, "w") as f:
        json.dump(tokens_db, f)
    return RedirectResponse(
        url=f"{redirect_uri}?code={code}&state={state}", status_code=303
    )


@app.post("/auth", response_class=HTMLResponse, include_in_schema=False)
def authorize_user(
    username: str = Form(...),
    password: str = Form(...),
    action: str = Form(...),
    redirect_uri: str = Form(...),
    state: str = Form(...),
):
    if action == "register":
        if username in users_db:
            return HTMLResponse("Username already registered", status_code=400)
        hashed_password = pwd_context.hash(password)
        users_db[username] = {
            "username": username,
            "password": hashed_password,
        }
        with open(users_pathname, "w") as f:
            json.dump(users_db, f)
        favorites_db[username] = []
        with open(favorites_pathname, "w") as f:
            json.dump(favorites_db, f)
        # Automatically log in the user after registration
        return login(username, redirect_uri, state)
    elif action == "login":
        user = users_db.get(username)
        if not user or not pwd_context.verify(password, user["password"]):
            return HTMLResponse(
                "Incorrect username or password", status_code=401
            )
        return login(username, redirect_uri, state)


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    username = tokens_db[code]
    encoded = jwt.encode(
        {"username": username}, key=PINECONE_API_KEY, algorithm="HS256"
    )
    jwts_db[username] = encoded
    with open(jwts_pathname, "w") as f:
        json.dump(jwts_db, f)
    return {"access_token": encoded}
