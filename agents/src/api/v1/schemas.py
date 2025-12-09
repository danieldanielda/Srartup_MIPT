from pydantic import BaseModel

class BarcodeRequest(BaseModel):
    barcode: str
    
class BarcodeResponse(BaseModel):
    barcode: str
    product_info: str