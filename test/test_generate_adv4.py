from dataset import Dataset
from utils import generate_adv4
from word_sim_task.config import ori_thesauri

dataset = Dataset()
dataset.load_task_datasets(*['SIMLEX999','SIMVERB3000-test','SIMVERB500-dev'])
generate_adv4(0.1,ori_thesauri,dataset.tasks)