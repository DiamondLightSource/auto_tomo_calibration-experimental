import numpy as np

centres = [[(550, 577), (592, 477)],
           [(870, 1219), (890, 558), (954, 896)],
           [(852, 590), (973, 907)],
           [(967, 886)],
           [(967, 885)],
           [(967, 886)],
           [(967, 886)]]

centroids_sphere = [[(123, 113), (147, 166)],
                    [(157, 97), (117, 241), (41, 74)],
                    [(83, 279), (70, 136)],
                    [(82, 83)],
                    [(97, 97)],
                    [(108, 109)],
                    [(117, 118)]]

N = len(centres)

index_top = [] # indexes of tops of spheres
centres_top = [] # centres of tops of spheres
index_bot = [] # indexes of bottoms of spheres
centres_bot = [] # centres of bottoms of spheres
centres_picked = [] # to store the (x,y) centres of spheres


for centre in centres[0]:
    centres_picked.append(centre)
    index_top.append(0)
    centres_top.append(centre)
    
# Process
for slice in range(1,N-2):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_fol in centres[slice+1]:
            # if centres equal to within about 10 pixels
            if np.allclose(np.asarray(centre), np.asarray(centre_fol), 0, 10): 
                break
        else:
            for centre_foll in centres[slice+2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_foll), 0, 10):
                    break
            # if centres are not the same within two neighbouring
            # slices then add them to centre bot
            else:
                index_bot.append(slice)
                centres_bot.append(centre)
        # If the centre has not been picked before: top of sphere
        # If the neighbouring slices have centres close together
        # then add them to centres picked and top arrays
        for centre_p in centres_picked:
            if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 10):
                break
        else:
            centres_picked.append(centre)
            index_top.append(slice)
            centres_top.append(centre)

for centre in centres[N-1]:
    index_bot.append(N-1)
    centres_bot.append(centre)

# Remove bugs = wrong centres that were detected only once
index_top_del = []
index_bot_del = []
for i_top in range(len(index_top)):
    if (index_top[i_top] in index_bot) and (centres_top[i_top] in centres_bot): # bugs are the only centres that appear once in both lists of edges
        i_bot = centres_bot.index((centres_top[i_top]))
        index_top_del.append(i_top)
        index_bot_del.append(i_bot)
        
index_top[:] = [item for i,item in enumerate(index_top) if i not in index_top_del]
centres_top[:] = [item for i,item in enumerate(centres_top) if i not in index_top_del]
index_bot[:] = [item for i,item in enumerate(index_bot) if i not in index_bot_del]
centres_bot[:] = [item for i,item in enumerate(centres_bot) if i not in index_bot_del]