## Converting from a pytorch .pt network to a .cc and .h file to run with tflite-micro

Install requirements.txt

pip install -r requirements.txt

update the convert_pytorch.py script to mach your model and model name, the 4 places to update are marked with TODO

Run this command to make it a .cc file so that it can run on a microcontroller without a file structure. TODO, update to you model name

xxd -i simple_net.tflite > simple_net.cpp

Then create a .hpp file that includes your network and the size. See simple_net.cpp and simple_net.hpp and the 3 TODO's

Now copy your .cpp and .hpp files over to PX4 and you are ready to use them