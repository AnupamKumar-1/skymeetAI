let IS_PROD = false;
const server = IS_PROD ?
    "https://skymeetai-backend.onrender.com" :

    "http://localhost:8000"


export default server;