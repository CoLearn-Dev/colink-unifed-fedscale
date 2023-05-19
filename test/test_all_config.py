import glob
import json
import pytest


import colink as CL


def simulate_with_config(config_file_path):
    from unifed.frameworks.fedscale.protocol import pop, UNIFED_TASK_DIR
    case_name = config_file_path.split("/")[-1].split(".")[0]
    with open(config_file_path, "r") as cf:
        config = json.load(cf)
    # use instant server for simulation
    ir = CL.InstantRegistry()
    config_participants = config["deployment"]["participants"]
    cls = []
    participants = []
    for p in config_participants:  # given user_ids are omitted and we generate new ones here
        role = p["role"]
        cl = CL.InstantServer().get_colink().switch_to_generated_user()
        pop.run_attach(cl)
        participants.append(CL.Participant(user_id=cl.get_user_id(), role=role))
        cls.append(cl)
    task_id = cls[0].run_task("unifed.fedscale", json.dumps(config), participants, True)
    cl.wait_task(task_id)

    example_log_from_server = cls[0].read_entry(f"unifed:task:{task_id}:log")
    print("Log from server:")
    print(example_log_from_server)
    example_log_from_client_0 = cls[1].read_entry(f"unifed:task:{task_id}:log")
    print("Log from client 0:")
    print(example_log_from_client_0)
    example_log_from_client_1 = cls[2].read_entry(f"unifed:task:{task_id}:log")
    print("Log from client 1:")
    print(example_log_from_client_1)

    example_output_from_server = cls[0].read_entry(f"unifed:task:{task_id}:output")
    print("Output from server:")
    print(example_output_from_server)
    example_output_from_client_0 = cls[1].read_entry(f"unifed:task:{task_id}:output")
    print("Output from client 0:")
    print(example_output_from_client_0)
    example_output_from_client_1 = cls[2].read_entry(f"unifed:task:{task_id}:output")
    print("Output from client 1:")
    print(example_output_from_client_1)


    return 


def test_load_config():
    # load all config files under the test folder
    config_file_paths = glob.glob("test/configs/*.json")
    assert len(config_file_paths) > 0


@pytest.mark.parametrize("config_file_path", glob.glob("test/configs/*.json"))
def test_with_config(config_file_path):
    if "skip" in config_file_path:
        pytest.skip("Skip this test case")
    results = simulate_with_config(config_file_path)
    assert all([r["error"] is None and r["return"]["returncode"] == 0 for r in results[1].values()])


if __name__ == "__main__":
    from pprint import pprint
    import time
    nw = time.time()
    target_case = "test/configs/case_0.json"
    simulate_with_config(target_case)
    print("Time elapsed:", time.time() - nw)