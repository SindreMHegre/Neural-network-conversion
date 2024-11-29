import torch
import torch.nn as nn
import ai_edge_torch
from networks import ControlModel


def convert_and_test_network():
    # Load the control model from aerialgym
    control_model = ControlModel([15, 256, 128, 6])
    control_model.load_state_dict(torch.load("big_quadBug_control.pt"))
    control_model.eval()

    # Use test inputs from aerialgym
    input_data = []
    expected_output = []
    input_data.append(torch.tensor([[0.14983742, 0.08736876, 0.41753131, 0.40901637, 0.81793004, 0.40459383, -0.84911108, 0.17873803, 0.49705449, 0.36090830, -0.34516647, 0.96046799, 0.51223969, -0.37822920, -0.82241392]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.20371765, 0.13641953, -0.98252934, 0.84184235, 0.86325157, 0.51837951]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.30168912, 0.03708152, 1.31934476, -0.92908216, -0.35560787, 0.10173219, 0.35931945, -0.93299562, 0.02021704, 0.00045894, -0.14975649, -1.63165092, -1.13141418, 0.89587283, 0.16081977]], dtype=torch.float32))
    expected_output.append(torch.tensor([[0.02822164, 0.47334626, 0.99998993, 0.34581655, -0.37028241, -0.31543487]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.46699604, -0.50998563, 0.16672820, 0.86817235, -0.46749234, -0.16651672, 0.37306786, 0.83608562, -0.40222049, 0.06366818, 0.26148084, -1.51914549, 1.38827813, -2.53781509, 0.21589912]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.63387084, 0.09936423, 0.99994993, -0.73621482, 0.72450620, -0.63493919]], dtype=torch.float32))

    input_data.append(torch.tensor([[0.19014239, 0.15077132, 0.14661583, 0.79394192, -0.53573620, -0.28747693, 0.56708807, 0.82301980, 0.03239720, -0.48295921, 0.34673548, -1.12633777, -0.70068616, 0.48624352, 0.80105269]], dtype=torch.float32))
    expected_output.append(torch.tensor([[0.24202859, 0.48122919, 0.99837452, 0.65230572, 0.44767538, -0.41530386]], dtype=torch.float32))

    input_data.append(torch.tensor([[0.24136041, -0.31392357, -0.31087670, 0.90179086, -0.23001933, 0.36587495, 0.42468345, 0.62857652, -0.65156400, 0.23377578, -0.07852923, -1.06731832, -2.33192945, -0.06323836, -0.98257381]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.67052197, -0.26490065, 0.99999762, -0.40146872, -0.06877456, 0.31986251]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.21389055, 0.48720673, 0.42695457, -0.21250533, 0.97169566, -0.10319418, -0.97498542, -0.20380539, 0.08869484, 0.65052676, 0.56230992, -1.64643896, 3.85724688, 1.40328085, 1.73034275]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.43670487, 0.64992654, 0.99997556, -0.81528223, -0.25585693, -0.20503660]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.57031095, -0.02563543, 0.33031416, 0.93534112, 0.21841492, -0.27826589, -0.19655465, 0.97490460, 0.10453337, 0.18247291, -0.28368610, -0.59923446, 1.00979388, -2.28870940, 0.99492550]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.08638971, 0.27503753, 0.99977303, -0.61759722, 0.71702272, -0.66452646]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.09258360, -0.21418959, 0.20377164, 0.87279713, 0.48532706, 0.05179608, -0.33098796, 0.51054299, 0.79359496, -0.09419722, -0.89054322, 0.66313821, 1.48728621, -0.31052846, 0.02000160]], dtype=torch.float32))
    expected_output.append(torch.tensor([[0.02509915, 0.40999347, -0.99392527, -0.42501256, 0.70090735, 0.23490459]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.12632474, -0.72671920, -0.01157777, 0.75734508, -0.63385677, -0.15701611, 0.44250205, 0.67496932, -0.59043062, 0.53840458, 0.63041091, -1.51975322, 1.39145863, -5.02927494, 0.30860049]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.62228310, 0.26200303, 0.99989682, -0.77313197, 0.91416949, -0.64553177]], dtype=torch.float32))

    input_data.append(torch.tensor([[-0.50845039, -1.40482628, 0.89889628, -0.77837610, 0.62778997, 0.00321817, -0.52914715, -0.65329641, -0.54148602, -0.15656121, -0.06492139, -1.37410843, 1.01404452, -3.13272572, -0.19120449]], dtype=torch.float32))
    expected_output.append(torch.tensor([[-0.82473534, 0.33078104, 0.99999058, 0.23091461, 0.52654755, 0.03732388]], dtype=torch.float32))

    sample_input = (torch.rand(1, 15),)
    tfLite_model = ai_edge_torch.convert(control_model, sample_input)
    tfLite_model.export('./big_control_net.tflite')

    output = []
    tfLite_output = []
    for i in range(len(input_data)):
        output.append(control_model(input_data[i]))
        tfLite_output.append(tfLite_model(*input_data[i]))

    for i in range(len(input_data)):
        print("Input: [" + ", ".join(f"{x:.8f}" for x in input_data[i].flatten()) + "]")
        print("Expected output: [" + ", ".join(f"{x:.8f}" for x in expected_output[i].flatten()) + "]")
        print("Actual output: [" + ", ".join(f"{x:.8f}" for x in output[i].flatten()) + "]")
        print("TFLite output:" + str(tfLite_output[i]))

        print("")

        with open("big_test_result.txt", "a") as f:
            f.write("[" + ", ".join(f"{x:.8f}" for x in output[i].flatten()) + "]\n")

    with open("big_test_result.txt", "a") as f:
        f.write("\n tfLite model output \n")
        for i in range(len(input_data)):
            f.write("[" + ", ".join(f"{x:.8f}" for x in tfLite_output[i].flatten()) + "]\n")

if __name__ == "__main__":
    convert_and_test_network()
