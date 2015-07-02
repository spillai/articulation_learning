from cylcmlogreader import PyLCMLogReader;
player = PyLCMLogReader();
player.init('/home/spillai/data/2013_articulation_test/book_unfolding/lcmlog-2013-03-16.00');

for utime in feature_utimes:
    img, cloud = player.getFrame(utime)
    

# # # ===== Video Animation ====
# image = None
# def animate(idx):
#     global player, image
#     if idx % 10 == 0: print 'Frame: ', idx
#     img,cloud = player.getFrame(feature_utimes[idx])
#     utime_text.set_text('Time: %4.2f' % feature_utimes[idx])
#     if image is None:
#         image = ax.imshow(img, interpolation='nearest', animated=True, label='video')
#     else:
#         image.set_data(img)
#         plt.draw()
#     return image,utime_text


# frames = feature_utimes.size
# anim = animation.FuncAnimation(fig, animate, frames=100, blit=True)
# anim.save('test.mp4', fps=20)
#plt.show()



# if True:
#     imcount = 0
#     path = '/home/spillai/data/2013_articulation_test/book_unfolding/processed.avi'
#     fourcc = cv2.VideoWriter_fourcc('i','Y','U','V')
#     writer = cv2.VideoWriter(path, fourcc, 1.0, (640, 480), True)
#     if writer: print 'writing'
#     for utime in feature_utimes:
#         img,cloud = player.getFrame(utime)
#         imcount += 1
#         #print img
#         writer.write(img)
#         if imcount > 10: break
#     writer.release()
#     print 'Done writing to ', path

    
# print feature_utimes.size
# feature_utimes

# for utime in feature_utimes:
#     img = 
    
# def init_frame():
#     """init animation"""
#     utime_text.set_text('')
#     return utime_text

# def animate(utime):
#     ""perform animation step""
#     utime_text.set_text('time = %7.3f' % utime)
#     return utime_text
