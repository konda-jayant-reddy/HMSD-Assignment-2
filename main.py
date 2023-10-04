from os import path

from fastapi import FastAPI, Request, File, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from scripts.RWT_AT_model import train_wt_at, predict_wt_at
from scripts.dissolved_oxygen import calculate_and_save_dissolved_oxygen

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# base routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/rwt", response_class=HTMLResponse)
async def rwt(request: Request):
    return templates.TemplateResponse(
        "rwt.html",
        {
            "request": request,
            "redirect": "redirect" in request.query_params.keys(),
        },
    )


@app.get("/do", response_class=HTMLResponse)
async def do(request: Request):
    if request.query_params.get("redirected") == "True":
        if path.exists("static/predicted_wt_daily.csv"):
            output_file_path = calculate_and_save_dissolved_oxygen(
                "static/predicted_wt_daily.csv"
            )
            return templates.TemplateResponse(
                "do.html", {"request": request, "output_file_path": output_file_path}
            )

    return templates.TemplateResponse("do.html", {"request": request})


# partial routes
@app.post("/results/rwt", response_class=HTMLResponse)
async def rwt_results(request: Request):
    options = await request.json()

    # train and predict model
    metrics = train_wt_at(
        model_type=int(options["model"]), perf_type=int(options["metric"])
    )
    predict_wt_at(model_type=int(options["model"]))

    # input data dictionary
    input_data_dict = {"air-temperature": "Air Temperature"}

    # observed data dictionary
    observed_data_dict = {"river-water-temperature": "River Water Temperature"}

    # model dictionary
    model_dict = {
        "0": "Linear Regression",
        "1": "Decision Tree Regression",
        "2": "Random Forest Regression",
    }

    # metric dictionary
    metric_dict = {"0": "RMSE", "1": "MAE", "2": "MSE"}

    response = templates.TemplateResponse(
        "partial/rwt_results.html",
        {
            "request": request,
            "input_data": input_data_dict[options["input-data"]],
            "observed_data": observed_data_dict[options["observed-data"]],
            "model": model_dict[options["model"]],
            "metric": metric_dict[options["metric"]],
            "train_error": metrics["train_error"],
            "test_error": metrics["test_error"],
        },
    )

    if request.query_params.get("redirect") == "True":
        response.headers["HX-Redirect"] = "/do?redirected=True"

    return response


@app.post("/results/do")
async def do_api(file: UploadFile):
    with open("static/predicted_wt_daily.csv", "wb") as f:
        f.write(await file.read())

    response = Response()
    response.headers["HX-Redirect"] = "/do?redirected=True"
    return response
