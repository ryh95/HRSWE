from pathlib import Path

exp_name = 'hrswe'
exp_res_dir = Path(exp_name+'_'+str(1))
exp_res_dir.mkdir(parents=True, exist_ok=True)
Path('antonyms.txt').rename(exp_res_dir / 'antonyms.txt')