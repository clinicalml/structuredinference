import fcntl,errno,time
with open('remove.me','a') as f:
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError as e:
            if e.errno != errno.EAGAIN:
                raise
            else:
                time.sleep(0.1)
    f.write('another line\n')
    fcntl.flock(f, fcntl.LOCK_UN)
