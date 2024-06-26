/*!
 * coding: utf-8
 * Author: Seunghyeon Kim
 */

const img = document.getElementById('video_frame');
const ws = new WebSocket('ws://172.27.1.11:8888/inference/det');
ws.binaryType = 'blob';

// 프레임 이미지를 갱신한다.
ws.onmessage = function (event) {
  const url = URL.createObjectURL(event.data);
  img.src = url;
};
// 에러 발생 시 콘솔 출력한다. (안 볼 것 같은데...?)
ws.onerror = function (event) {
  console.error('websocket error:', event);
};
// 브라우저 탭 닫았을 때 웹소켓 연결 해제 신호를 서버에 보낸다.
// 추론 서버의 자원 절약을 위해!
window.addEventListener('beforeunload', () => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.close(1000, 'client closed the connection.');
  }
});
// 웹소켓 닫힌 후 후처리(로깅).
ws.onclose = function (event) {
  if (!event.wasClean) {
    console.error(`websocket close with some error: ${event.reason}`);
  }
  else {
    console.log('websocket closed cleanly.')
  }
};

const chk = document.getElementById('resize_toggle');
chk.addEventListener('change', () => {
  ws.send(JSON.stringify(
    { resize: chk.checked }
  ));
});
