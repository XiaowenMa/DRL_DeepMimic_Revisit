from DRL import DRL
import tqdm

if __name__=="__main__":
    drl = DRL()
    for i in tqdm.tqdm(range(10)):
        drl.rollout(i)
        drl.update()