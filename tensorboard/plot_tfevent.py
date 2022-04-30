import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
from scipy.signal import savgol_filter  # for smoothing

path_to_events_file = 'tensorboard/base_encoder/events.out.tfevents.1651255328.DESKTOP-DP8CJ8F.16040.0'


logs = [summary for summary in summary_iterator(path_to_events_file)]
lods_dict = {
    'Loss/train': [],
    'Loss/valid': []
}
for log in logs:
    try:
        lods_dict[log.summary.value[0].tag].append(log.summary.value[0].simple_value)
    except:
        pass

print(len(lods_dict['Loss/train']))
print(len(lods_dict['Loss/valid']))

# smoothing
lods_dict['Loss/train'] = savgol_filter(lods_dict['Loss/train'], len(lods_dict['Loss/train'])//5+1, 2)
lods_dict['Loss/valid'] = savgol_filter(lods_dict['Loss/valid'], len(lods_dict['Loss/valid'])//5+1, 2)

# plot loss/train and loss/valid on two subplots
fig, ax = plt.subplots(2, 1)
ax[0].plot(lods_dict['Loss/train'])
ax[0].set_title('Loss/train')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')

ax[1].plot(lods_dict['Loss/valid'])
ax[1].set_title('Loss/valid')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Loss')

plt.show()
fig.savefig('images/base_detector_curves.png')
