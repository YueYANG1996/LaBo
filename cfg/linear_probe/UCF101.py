steps = 8
n_runs = 1
img_path = 'datasets/UCF101/images/'
data_root = 'exp/linear_probe/UCF101'
img_split_path = 'datasets/UCF101/splits'

num_cls = 101
unfreeze_clip = False
paper = True

cls_names = ['apply eye makeup', 'apply lipstick', 'archery', 'baby crawling', 'balance beam', 'band marching', 'baseball pitch', 'basketball', 'basketball dunk', 'bench press', 'biking', 'billiards', 'blow dry hair', 'blowing candles', 'body weight squats', 'bowling', 'boxing punching bag', 'boxing speed bag', 'breast stroke', 'brushing teeth', 'clean and jerk', 'cliff diving', 'cricket bowling', 'cricket shot', 'cutting in kitchen', 'diving', 'drumming', 'fencing', 'field hockey penalty', 'floor gymnastics', 'frisbee catch', 'front crawl', 'golf swing', 'haircut', 'hammering', 'hammer throw', 'handstand pushups', 'handstand walking', 'head massage', 'high jump', 'horse race', 'horse riding', 'hula hoop', 'ice dancing', 'javelin throw', 'juggling balls', 'jumping jack', 'jump rope', 'kayaking', 'knitting', 'long jump', 'lunges', 'military parade', 'mixing', 'mopping floor', 'nunchucks', 'parallel bars', 'pizza tossing', 'playing cello', 'playing daf', 'playing dhol', 'playing flute', 'playing guitar', 'playing piano', 'playing sitar', 'playing tabla', 'playing violin', 'pole vault', 'pommel horse', 'pull ups', 'punch', 'push ups', 'rafting', 'rock climbing indoor', 'rope climbing', 'rowing', 'salsa spin', 'shaving beard', 'shotput', 'skate boarding', 'skiing', 'skijet', 'sky diving', 'soccer juggling', 'soccer penalty', 'still rings', 'sumo wrestling', 'surfing', 'swing', 'table tennis shot', 'tai chi', 'tennis swing', 'throw discus', 'trampoline jumping', 'typing', 'uneven bars', 'volleyball spiking', 'walking with dog', 'wall pushups', 'writing on board', 'yo yo']
img_ext=''
clip_model='ViT-L/14'

lr=1e-3
bs=128
n_shots = 8
dataset = 'UCF101'