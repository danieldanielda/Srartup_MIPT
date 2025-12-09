import logging

from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from crew import BarcodeLookupCrew

from src.api.v1.schemas import BarcodeRequest, BarcodeResponse

router = APIRouter()

@router.post("/search_barcode")
async def search_barcode(request: BarcodeRequest) -> BarcodeResponse:
    barcode = request.barcode.strip()
    if not barcode.isdigit():
        raise HTTPException(status_code=400, detail="Barcode must contain only digits")

    try:

        inputs = {"barcode": barcode}
        result = BarcodeLookupCrew().crew().kickoff(inputs=inputs)

        response_result = BarcodeResponse(
            barcode=barcode,
            product_info=str(result)
        )
        return response_result
    except Exception as e:
        logging.error(f"Error during barcode lookup: {e}")
        raise HTTPException(status_code=500, detail="Failed to process barcode")