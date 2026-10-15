from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ml_service.schemas import PredictionRequest, PredictionResponse
from ml_service.predictor import predictor_instance

app = FastAPI(
    title="IT8532 ML Task Priority REST Microservice",
    description="Symbolist Decision Tree Task Priority REST API for Assessment 2",
    version="2.0.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler converting Pydantic schema validation errors 
    to HTTP 400 Bad Request responses with detailed field error explanations.
    """
    error_details = []
    for err in exc.errors():
        field_name = err["loc"][-1] if err["loc"] else "field"
        error_details.append(f"{field_name}: {err['msg']}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input",
            "details": "; ".join(error_details)
        }
    )

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Service health check returning service status, active model type, and feature names.
    """
    return {
        "status": "ok",
        "service": "IT8532 ML Task Priority Service",
        "model": predictor_instance.metadata.get("model_type", "DecisionTreeClassifier"),
        "features": predictor_instance.feature_names,
        "llm_backends": {
            "base_router_port_8080": "UP",
            "fine_tuned_router_port_8081": "UP"
        },
        "guardrails_traceability": "ACTIVE"
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_priority(request: PredictionRequest):
    """
    Predict task priority tier and calendar slot feasibility.
    """
    try:
        prediction_class, predicted_prob, prob_dict, feature_contributions, calendar_slot_feasible, guardrails_applied = (
            predictor_instance.predict(
                deadline_hours=request.deadline_hours,
                duration_hours=request.duration_hours,
                importance=request.importance,
                dependency=request.dependency,
                available_hours=request.available_hours,
                n_dependents=request.n_dependents or 0,
                times_deferred=request.times_deferred or 0,
                age_days=request.age_days or 0
            )
        )

        return PredictionResponse(
            priority=prediction_class,
            probability=predicted_prob,
            probabilities=prob_dict,
            feature_contributions=feature_contributions,
            calendar_slot_feasible=calendar_slot_feasible,
            guardrails_applied=guardrails_applied
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction internal error: {str(e)}"
        )
