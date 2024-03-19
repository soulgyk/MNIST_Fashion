# ML Ops Challenge

This project is a solution to the Machine Learning Ops Challenge.

## Project Structure

- `main.py`: Defines the FastAPI app and endpoint for inference.
- `Dockerfile`: Specifies Docker containerization instructions.
- `requirements.txt`: Lists required Python packages.

## Software and Tools Requirements

1. [Github Account](https://github.com)
2. [VSCodeIDE](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/docs/gitcli)
4. [Docker](https://docs.docker.com/get-docker/)


Create a new environment

```
conda create -p myenv python==3.7 -y

```
Activate the myenv environment

 ```
 conda activate myenv/

 ```


## Setup Instructions

1. Clone this repository.
2. Install Docker if not already installed: [Docker Installation Guide](https://docs.docker.com/get-docker/)
3. Build the Docker image:
    ```bash
    docker build -t ml-api .
    ```
4. Run the Docker container:
    ```bash
    docker run -d -p 8000:8000 ml-api
    ```

## Usage

- Send a POST request to `http://localhost:8000/predict/` with image data in the request body.
- Example request body:
    ```json
    {"images": [[0, 0, 255, ...], [255, 0, 0, ...], ...]}
    ```

## Additional Notes

- Ensure that Docker is running and accessible on your system.
- Modify the model initialization and preprocessing according to your requirements.


The input for testing is given below:

{
  "pixels": [
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 83, 142, 50, 0, 0, 0, 0, 85, 145, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 210, 208, 255, 254, 225, 227, 255, 221, 199, 211, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 105, 213, 187, 187, 204, 223, 230, 227, 221, 188, 183, 188, 188, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 206, 185, 193, 189, 230, 219, 229, 205, 180, 186, 181, 201, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 206, 214, 190, 185, 177, 204, 244, 215, 174, 181, 177, 187, 209, 118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 196, 219, 178, 184, 183, 177, 222, 181, 173, 184, 173, 203, 210, 177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 211, 219, 83, 199, 197, 184, 201, 201, 185, 206, 153, 150, 223, 205, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 217, 220, 61, 205, 196, 188, 194, 211, 199, 203, 159, 112, 226, 194, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 222, 253, 0, 203, 197, 193, 185, 194, 204, 211, 155, 73, 233, 203, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 234, 207, 0, 219, 201, 196, 207, 190, 194, 230, 105, 0, 255, 210, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 157, 243, 163, 0, 245, 203, 215, 209, 215, 182, 231, 142, 0, 255, 223, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 241, 142, 0, 230, 192, 234, 198, 236, 199, 203, 144, 0, 228, 222, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 251, 132, 52, 236, 191, 204, 182, 236, 210, 190, 226, 0, 216, 240, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 146, 223, 87, 132, 223, 192, 196, 186, 215, 201, 184, 231, 55, 122, 218, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 146, 251, 77, 144, 223, 188, 202, 199, 217, 193, 184, 244, 54, 161, 215, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 255, 60, 203, 220, 194, 194, 208, 201, 192, 198, 242, 73, 198, 255, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 242, 137, 243, 212, 211, 213, 205, 213, 221, 212, 181, 238, 3, 190, 253, 91, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 213, 255, 0, 86, 255, 217, 228, 227, 216, 255, 0, 120, 254, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 255, 0, 219, 247, 221, 225, 216, 206, 255, 124, 68, 255, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 229, 234, 0, 255, 227, 228, 230, 228, 221, 249, 198, 18, 255, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 185, 6, 255, 220, 231, 230, 223, 217, 241, 221, 2, 255, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 237, 157, 29, 255, 220, 230, 232, 223, 218, 235, 237, 0, 255, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235, 138, 37, 255, 221, 231, 231, 232, 229, 233, 248, 0, 255, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 138, 36, 255, 227, 233, 230, 225, 214, 230, 252, 0, 255, 157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 24, 255, 228, 233, 232, 234, 230, 233, 245, 0, 61, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 213, 234, 230, 228, 225, 222, 239, 196, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 163, 193, 224, 218, 201, 230, 228, 85, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 206, 252, 239, 243, 243, 233, 250, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 163, 193, 224, 218, 201, 230, 228, 85, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 147, 80, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 24, 88, 66, 7, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 27, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 26, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 70, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
}