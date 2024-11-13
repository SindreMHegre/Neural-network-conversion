import torch
import torch.nn as nn
import ai_edge_torch

class fullNetwork(nn.Module):
    def __init__(self):
        super(fullNetwork, self).__init__()
        self.fc1 = nn.Linear(15, 32)  # Input layer (15 inputs, position error, velocity, attitude (6D), angular velocity)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 6)
        self.fc4 = nn.Linear(6, 32)
        self.fc5 = nn.Linear(32, 4) # Output layer (number of motors)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class controlNetwork(nn.Module):
    def __init__(self):
        super(controlNetwork, self).__init__()
        self.fc1 = nn.Linear(15, 32)  # Input layer (15 inputs, position error, velocity, attitude (6D), angular velocity)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 6) # Output layer (thrusts and torques)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class allocationNetwork(nn.Module):
    def __init__(self):
        super(allocationNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # Input layer (thrusts and torques)
        self.fc2 = nn.Linear(32, 4) # Output layer (number of motors)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ControlModel(nn.Module):
    def __init__(self, layer_sizes):
        super(ControlModel, self).__init__()
        self.control_stack = nn.ModuleList([])

        self.control_stack.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
        if len(layer_sizes) > 2:
            self.control_stack.append(nn.Tanh())
            for i, input_size in enumerate(layer_sizes[1:-1]):
                output_size = layer_sizes[i+2]
                self.control_stack.append(
                    nn.Linear(input_size, output_size).to(torch.float)).cpu()
                self.control_stack.append(nn.Tanh())

    def forward(self, x):
        for l_or_a in self.control_stack:
            x = l_or_a(x)

        return x

class AllocationModel(nn.Module):
    def __init__(self, layer_sizes):
        super(AllocationModel, self).__init__()
        self.allocation_stack = nn.ModuleList([])

        self.allocation_stack.append(nn.Linear(layer_sizes[-3], layer_sizes[-2]).to(torch.float)).cpu()
        self.allocation_stack.append(nn.ReLU())
        self.allocation_stack.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]).to(torch.float)).cpu()

    def forward(self, x):
        for l_or_a in self.allocation_stack:
            x = l_or_a(x)

        return x


def split_network_jit(full_model_path):
    # Load the full network
    full_model = torch.jit.load(full_model_path)

    control_stack = getattr(full_model, 'control_stack')
    allocation_stack = getattr(full_model, 'allocation_stack')

    # Extract control network
    control_model = controlNetwork()
    control_model.fc1.weight.data = getattr(control_stack, '0').weight.data.clone()
    control_model.fc1.bias.data = getattr(control_stack, '0').bias.data.clone()
    control_model.fc2.weight.data = getattr(control_stack, '2').weight.data.clone()
    control_model.fc2.bias.data = getattr(control_stack, '2').bias.data.clone()
    control_model.fc3.weight.data = getattr(control_stack, '4').weight.data.clone()
    control_model.fc3.bias.data = getattr(control_stack, '4').bias.data.clone()

    # Extract allocation network
    allocation_model = allocationNetwork()
    allocation_model.fc1.weight.data = getattr(allocation_stack, '0').weight.data.clone()
    allocation_model.fc1.bias.data = getattr(allocation_stack, '0').bias.data.clone()
    allocation_model.fc2.weight.data = getattr(allocation_stack, '2').weight.data.clone()
    allocation_model.fc2.bias.data = getattr(allocation_stack, '2').bias.data.clone()

    # Save the control network
    torch.save(control_model.state_dict(), "control_net.pt")

    # Save the allocation network
    torch.save(allocation_model.state_dict(), "allocation_net.pt")


# TODO, the full network does not work
def test_split_networks(self_split: bool = True):
    # Load the full network
    #full_model = torch.jit.load("quadBug.pt")
    # full_model = fullNetwork()
    # full_model.load_state_dict(torch.load("quadBug.pth"))
    # full_model.eval()
    full_model = torch.jit.load("quadBug.pt")

    if self_split:
        control_model = ControlModel([15, 32, 24, 6])
        control_model.load_state_dict(torch.load("quadBug_control.pt"))
        # allocation_model = AllocationModel([6, 32, 4])
        # allocation_model.load_state_dict(torch.load("quadBug_allocation.pt"))
        #control_model = torch.jit.load("quadBug_control_jit.pt")
        allocation_model = torch.jit.load("quadBug_allocation_jit.pt")
    else:
        # Load the split networks
        control_model = controlNetwork()
        control_model.load_state_dict(torch.load("control_net.pth"))
        allocation_model = allocationNetwork()
        allocation_model.load_state_dict(torch.load("allocation_net.pth"))

    control_model.eval()
    allocation_model.eval()

    # Use test inputs from aerialgym
    # input_data = torch.tensor([[ 0.0425,  0.0919, -0.1422, -0.8354,  0.5439,  0.0797, -0.5426, -0.8391,
    #      0.0385,  0.1862, -0.0112, -0.0232,  0.1448, -0.3787,  0.0386]], dtype=torch.float32)
    # expected_thrust_torque = torch.tensor([[-0.4832,  0.6225,  0.7115, -0.0784,  0.9561, -0.3821]], dtype=torch.float32)
    # expected_motor_commands = torch.tensor([[0.8645, 1.1000, 1.1000, 0.5888]], dtype=torch.float32)
    input_data = torch.tensor([[ 0.3241, -0.4019, -0.1132,  0.6828, -0.7281,  0.0602,  0.7296,  0.6838,
        -0.0053, -0.3517, -0.2642,  0.1048, -0.5758,  0.8968,  0.0043]], dtype=torch.float32)
    expected_thrust_torque = torch.tensor([[-0.0740, -0.2835,  0.5911,  0.9189, -0.6165, -0.2095]], dtype=torch.float32)
    expected_motor_commands = torch.tensor([[0.9492, 0.5000, 1.1000, 1.1000]], dtype=torch.float32)

    #without scaling
    # input_data = torch.tensor([[ 0.1341,  0.1014,  0.3978,  0.3807,  0.8522,  0.3589, -0.8158,  0.1267,
    #      0.5643,  0.3625, -0.3390,  0.8639,  0.5186, -0.4435, -0.8138]], dtype=torch.float32)
    # expected_thrust_torque = torch.tensor([[-0.2037,  0.1364, -0.9825,  0.8418,  0.8633,  0.5184]], dtype=torch.float32)
    # expected_motor_commands = torch.tensor([[0.5000, 0.5000, 1.1000, 0.8021]], dtype=torch.float32)

    full_output = full_model(input_data)

    # Run the split networks
    control_output = control_model(input_data)
    allocation_output = allocation_model(control_output)
    allocation_output_given = allocation_model(expected_thrust_torque)

    #rescaled_output = rescale_actions(control_output)
    #rescaled_allocation_output = allocation_model(rescaled_output)

    sample_input = (torch.rand(1, 15),)
    tfLite_model = ai_edge_torch.convert(control_model, sample_input)
    tfLite_output = tfLite_model(*input_data)
    tfLite_model.export('./control_net.tflite')

    # Compare the outputs
    print("Input:" + str(input_data))
    print("Expected thrust and torque:" + str(expected_thrust_torque))
    print("Actual thrust and torque:" + str(control_output))
    print("TFLite output:" + str(tfLite_output))
    #print("Rescaled output:" + str(rescaled_output))
    #print("Expected motor commands:" + str(expected_motor_commands))
    #print("Actual motor commands:" + str(allocation_output))
    #print("Allocation output given thrust and torque:" + str(allocation_output_given))
    #print("Rescaled allocation output:" + str(rescaled_allocation_output))
    #print("Full output:" + str(full_output))


def rescale_actions(scaled_command_actions):
    max_thrust = torch.tensor([0.0, 0.0, 4.4])
    min_thrust = torch.tensor([0.0, 0.0, 2.0])
    max_torque = torch.tensor([0.096, 0.096, 1.2])
    min_torque = torch.tensor([-0.096, -0.096, -1.2])
    command_actions = scaled_command_actions.clone()

    command_actions[:, :3] = scaled_command_actions[:, :3] * (max_thrust - min_thrust)/2 + (max_thrust + min_thrust)/2
    command_actions[:, 3:] = scaled_command_actions[:, 3:] * (max_torque - min_torque)/2 + (max_torque + min_torque)/2

    return command_actions


def print_model_structure(model, indent=0):
    for name, module in model.named_children():
        print(' ' * indent + name + ': ' + str(module))
        print_model_structure(module, indent + 2)


if __name__ == "__main__":
    #split_network("quadBug.pth")
    test_split_networks()


    # From aerial gym convert_model.py:

# class ModelDeploy(nn.Module):
#     def __init__(self, layer_sizes, lims):
#         super(ModelDeploy, self).__init__()
#         self.control_stack = nn.ModuleList([])
#         self.allocation_stack = nn.ModuleList([])

#         self.max_thrust = lims["max_thrust"]
#         self.min_thrust = lims["min_thrust"]
#         self.max_torque = lims["max_torque"]
#         self.min_torque = lims["min_torque"]
#         self.max_u = lims["max_u"]
#         self.min_u = lims["min_u"]

#         # control layers
#         self.control_stack.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
#         if len(layer_sizes) > 2:
#             self.control_stack.append(nn.Tanh())
#             for i, input_size in enumerate(layer_sizes[1:-3]):
#                 output_size = layer_sizes[i+2]
#                 self.control_stack.append(
#                     nn.Linear(input_size, output_size).to(torch.float)).cpu()
#                 self.control_stack.append(nn.Tanh())

#         # allocation layers
#         self.allocation_stack.append(nn.Linear(layer_sizes[-3], layer_sizes[-2]).to(torch.float)).cpu()
#         self.allocation_stack.append(nn.ReLU())
#         self.allocation_stack.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]).to(torch.float)).cpu()

#     def rescale_actions(self, scaled_command_actions):
#         # rescale obtain wrenches
#         command_actions = scaled_command_actions.clone()

#         command_actions[:3] = scaled_command_actions[:3] * (self.max_thrust - self.min_thrust)/2 + (self.max_thrust + self.min_thrust)/2
#         command_actions[3:] = scaled_command_actions[3:] * (self.max_torque - self.min_torque)/2 + (self.max_torque + self.min_torque)/2

#         return command_actions


#     def forward(self, x):
#         for l_or_a in self.control_stack:
#             x = l_or_a(x)

#         x = self.rescale_actions(x)

#         for l_or_a in self.allocation_stack:
#             x = l_or_a(x)

#         return x


    # # Deployment of torch models for robotics applications:
    # # https://pytorch.org/blog/model-serving-in-pyorch/

    # max_thrust = torch.max(torch.tensor(robot_model.convex_hull_admissible_set_vertices[:,:3]), dim=0).values.to(torch.float).cpu()
    # min_thrust = torch.min(torch.tensor(robot_model.convex_hull_admissible_set_vertices[:,:3]), dim=0).values.to(torch.float).cpu()
    # max_torque = torch.max(torch.tensor(robot_model.convex_hull_admissible_set_vertices[:,3:]), dim=0).values.to(torch.float).cpu()
    # min_torque = torch.min(torch.tensor(robot_model.convex_hull_admissible_set_vertices[:,3:]), dim=0).values.to(torch.float).cpu()
    # max_u = torch.tensor(robot_model.max_u).to(torch.float).cpu()
    # min_u = torch.tensor(robot_model.min_u).to(torch.float).cpu()

    # lims = {"max_thrust":max_thrust, "min_thrust": min_thrust, "max_torque": max_torque, "min_torque": min_torque, "max_u": max_u, "min_u": min_u}

    # nn_model_deploy = ModelDeploy([15, 32, 24, 6, 32, robot_model.n_motors], lims)
    # allocation_model = torch.load(AERIAL_GYM_DIRECTORY + "/aerial_gym/control/learning_based_control_allocation/saved_models/" + robot_model.cfg_name + ".pth")

    # nn_model_deploy.control_stack[0].weight.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.obs.mlp_head[0].weight.data
    # nn_model_deploy.control_stack[0].bias.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.obs.mlp_head[0].bias.data
    # nn_model_deploy.control_stack[2].weight.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.obs.mlp_head[2].weight.data
    # nn_model_deploy.control_stack[2].bias.data[:] = nn_model_full.actor_critic.actor_encoder.encoders.obs.mlp_head[2].bias.data
    # nn_model_deploy.control_stack[4].weight.data[:] = nn_model_full.actor_critic.action_parameterization.distribution_linear.weight.data
    # nn_model_deploy.control_stack[4].bias.data[:] = nn_model_full.actor_critic.action_parameterization.distribution_linear.bias.data
    # nn_model_deploy.allocation_stack[0].weight.data[:] = allocation_model.stack[0].weight.data
    # nn_model_deploy.allocation_stack[0].bias.data[:] = allocation_model.stack[0].bias.data
    # nn_model_deploy.allocation_stack[2].weight.data[:] = allocation_model.stack[2].weight.data
    # nn_model_deploy.allocation_stack[2].bias.data[:] = allocation_model.stack[2].bias.data

    # sm = torch.jit.script(nn_model_deploy)
    # torch.jit.save(sm, "./deployment/deployed_models/" + robot_model.cfg_name + ".pt")

    # print('Size normal (B):', os.path.getsize("./deployment/deployed_models/" + robot_model.cfg_name + ".pt"))

    # return sm