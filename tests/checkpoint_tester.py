from latentvideodiffusion import utils
import dill 
from types import SimpleNamespace

def dummy_train(args,cfg,num_loops,print_state=False):

    state, checkpoint_state = utils.get_checkpoint_state(args,cfg)
    
    if state is None : 
        state = [0,0,0,0]
 

    for i in range(num_loops):
        state[3] = state[3] + 1
        print(f"\t Iteration : {i}")

        if print_state:
            print("\n Checkpoint List :")
            print(checkpoint_state[4])
            print(f"State: {state[3]}")
            
        checkpoint_state = utils.update_checkpoint_state(state, checkpoint_state)

test_args = {
    "checkpoint" : "/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests/load_checkpoint_test.pkl"
}





def main():

    N = 100
    load = False
  

    if load :
        test_args = SimpleNamespace(checkpoint = "/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests/test_state.pkl")
    else:
        test_args = SimpleNamespace(checkpoint = None)

    ckpt_params = {
        "checkpoints":{
            "ckpt_name" : "test",
            "ckpt_dir"  : "/home/vidmod/lvd-dev/lvd-work-dir/lvd-arjun/tests",
            "max_ckpts" : 5,
            "ckpt_interval" : 3
        }
    }

    dummy_train(test_args, ckpt_params,N,True)
    


if __name__ == "__main__":
    main()