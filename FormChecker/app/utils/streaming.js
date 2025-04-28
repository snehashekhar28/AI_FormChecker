import io from 'socket.io-client';
import EventEmitter from 'events';

// point this at your backend
const BACKEND_URL = 'http://172.20.10.9:5000';

// Create one socket, immediately
const socket = io(BACKEND_URL, {
  autoConnect: true,            // will connect right away
  reconnection: true,
});

const emitter = new EventEmitter();
let lastAnalysis = null;

socket.on('connect', () => console.log('Socket connected', socket.id));
socket.on('disconnect', () => console.log('Socket disconnected'));
socket.on('connect_error', (err) => console.error('Socket error', err));

socket.on('results', data => {
    lastAnalysis = data;
    emitter.emit('results', data);
});

export function subscribeAnalysis(callback) {
    // emit right away if we already have it
    if (lastAnalysis !== null) callback(lastAnalysis);
    emitter.on('results', callback);
    return () => emitter.off('results', callback);
}

export function resetLastAnalysis() {
    lastAnalysis = null;
}

export default socket;