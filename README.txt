Advanced Black Hole Lensing Simulator

An interactive, real-time relativistic ray-tracing simulator for black holes, featuring accurate Schwarzschild spacetime metrics, M87 Doppler beaming, and TON 618 quasar jets.

⚠️ Hardware & System Requirements

Because this simulation performs parallel raymarching across hundreds of thousands of pixels in real-time, it requires direct access to GPU hardware.

GPU: NVIDIA Graphics Card (AMD/Intel integrated graphics are not supported).

NVIDIA CUDA Toolkit: You must have the official NVIDIA CUDA Toolkit installed on your system.

C++ Compiler: PyCUDA compiles the physics kernel on the fly.

Windows Users: You must have Microsoft Visual Studio installed with the "Desktop development with C++" workload.

📦 Installation

Clone this repository to your local machine.

It is highly recommended to use a virtual environment:

python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


Install the required Python dependencies:

pip install -r requirements.txt


🚀 How to Run (Windows Users - IMPORTANT)

If you are on Windows, running this from a standard PowerShell or Command Prompt will likely result in a CompileError: nvcc preprocessing failed because Python cannot locate your C++ compiler.

To fix this:

Open your Windows Start Menu.

Search for and open the x64 Native Tools Command Prompt for VS.

Navigate to your project folder: cd path\to\your\folder

Activate your environment: venv\Scripts\activate.bat

Run the simulation:

python blackhole_sim_ton618.py


🎮 Controls

Click the window to ensure it has focus.

I: Open the Target Dialog box (Type M87 or TON 618).

J: Toggle Optical Filter (Only available in M87 mode).

W / S: Increase / Decrease Black Hole Mass.

Z / X: Zoom In / Zoom Out.

Up Arrow / Down Arrow: Adjust Camera Elevation/Tilt.

R: Reset simulation to default parameters.