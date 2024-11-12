import torch
import torch.nn as nn

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


def split_network(full_model_path):
    # Load the full network
    full_model = fullNetwork()
    full_model.load_state_dict(torch.load(full_model_path))

    # Extract control network
    control_model = controlNetwork()
    control_model.fc1 = full_model.fc1
    control_model.fc2 = full_model.fc2
    control_model.fc3 = full_model.fc3

    # Extract allocation network
    allocation_model = allocationNetwork()
    allocation_model.fc1 = full_model.fc4
    allocation_model.fc2 = full_model.fc5

    # Save the control network
    torch.save(control_model.state_dict(), "control_net.pt")

    # Save the allocation network
    torch.save(allocation_model.state_dict(), "allocation_net.pt")


# TODO, the full network does not work
def test_split_networks():
    # Load the full network
    full_model = torch.jit.load("quadBug.pt")
    full_model.eval()

    # Load the split networks
    control_model = controlNetwork()
    control_model.load_state_dict(torch.load("control_net.pt"))
    control_model.eval()

    allocation_model = allocationNetwork()
    allocation_model.load_state_dict(torch.load("allocation_net.pt"))
    allocation_model.eval()

    # Generate random input
    input_data = torch.randn(1, 15)

    # Run the full network
    with torch.no_grad():
        full_output = full_model(input_data)

    # Run the split networks
    control_output = control_model(input_data)
    allocation_output = allocation_model(control_output)

    # Compare the outputs
    diff = torch.abs(full_output - allocation_output)
    max_diff = torch.max(diff).item()

    print(f"Max difference between full network and split networks: {max_diff}")
    assert max_diff < 1e-5, "The outputs of the split networks do not match the full network"


def print_model_structure(model, indent=0):
    for name, module in model.named_children():
        print(' ' * indent + name + ': ' + str(module))
        print_model_structure(module, indent + 2)


if __name__ == "__main__":
    split_network_jit("quadBug.pt")
    # test_split_networks()


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