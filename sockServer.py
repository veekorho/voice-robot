import socket
import time
from threading import Timer
from threading import Thread
from threading import Lock
from queue import Queue 
import stt
import actualrealtime_stt2queue

# queue to get transcribed speech
raw_q = Queue()
send_q = Queue()

##Don't send data at the same time from different threads
send_lock = Lock()

###############################################


## CONNECT TO RASPBERRY PI
def connect():
    clientSocket, address = s.accept()
    with send_lock:
        clientSocket.sendall(bytes("connected\n", "utf-8"))
    return clientSocket, address


## SEND DATA TO RASPBERRY PI
def sendFromTerminal():
    while True:
        msg = input("Send message:") + "\n"
        with send_lock:
            clientSocket.sendall(bytes(msg, "utf-8"))

def send(send_q):
    while True:
        msg = send_q.get()
        print(msg)
        with send_lock:
            clientSocket.sendall(bytes(msg, "utf-8"))


## LISTEN FOR DATA FROM RASPBERRY PI (currently not really used, but could be :D) 
def listen():
    while True:
        data = clientSocket.recv(1024)
        if not data:
            print("Client disconnected.")
            break
        print("Recieved:", data.decode("utf-8"))


## TRANSCRIBE SPEECH TO GET COMMANDS
def getCommand(raw_q, send_q): 
    print("starting stt...")

    #start threads
    #listen_thread = Thread(target=stt.listen, daemon=True)
    transcribe_thread = Thread(target=actualrealtime_stt2queue.transcribe, args=(raw_q,), daemon=True)
    #transcribe_thread = Thread(target=stt.transcribe, args=(raw_q,), daemon=True)
    #listen_thread.start()
    transcribe_thread.start()

    #last_transcription = ""

    # put words only new heard words into the send queue
    #while True:
        #raw_transcription = raw_q.get()
        #for word in raw_transcription.split():
            #if word not in last_transcription:
                #send_q.put(word)
        #last_transcription = raw_transcription
    while True:
        msg = raw_q.get()
        send_q.put(msg)
            

################################################################

#connect before starting anything
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 6966))
s.listen(1)

print('Server is now running.')

clientSocket, address = connect()
print(f"Connection from {address} has been established.")

#make threads for different things
threads = []
threads.append(Thread(target=listen, daemon=True))                  #listen from raspberry
threads.append(Thread(target=send, args=(send_q,), daemon=True))         #send to raspberry
threads.append(Thread(target=getCommand, args=(raw_q, send_q), daemon=True))   #get commands from mic + send only nex words



try:
    for thread in threads:
        thread.start()
    while True:
        clientSocket, address = connect()
except KeyboardInterrupt:
    pass
finally:
    print("Shutdown...")      # are there any unterminated while loops?
    for thread in threads:
        thread.join()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    
    
