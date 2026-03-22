import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analyzer import analyze, AnalyzerError

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_race(request: Request, race_data: str = Form(...)):
    try:
        result = analyze(race_data)
        result_html = md.markdown(result, extensions=["tables", "nl2br"])
        return templates.TemplateResponse(
            "partials/analysis.html",
            {"request": request, "result": result_html},
        )
    except AnalyzerError as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(e)},
            status_code=200,  # Return 200 so HTMX swaps it in normally
        )
