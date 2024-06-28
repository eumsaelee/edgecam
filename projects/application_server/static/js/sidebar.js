/*!
 * coding: utf-8
 * Author: Seunghyeon Kim
 */

document.addEventListener("DOMContentLoaded", function() {
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", function(event) {
      event.preventDefault(); // 페이지 전체 리로드 방지

      const url = this.getAttribute("href");
      fetch(url)
        .then(response => response.text())
        .then(html => {
          const mainContent = document.getElementById("main-content");
          mainContent.innerHTML = html;

          // 페이지에 삽입된 모든 스크립트 태그를 찾아서 각각 처리
          const scripts = mainContent.querySelectorAll("script");
          scripts.forEach(oldScript => {
            const newScript = document.createElement("script");
            if (oldScript.src) {
              newScript.src = oldScript.src;
              newScript.onload = () => console.log(`Loaded script ${oldScript.src}`);
            } else {
              newScript.textContent = oldScript.textContent;
            }
            oldScript.parentNode.replaceChild(newScript, oldScript);
          });
        });

      // 링크들의 클래스 및 속성을 변경
      navLinks.forEach(lnk => {
        lnk.classList.remove('active');
        lnk.removeAttribute('aria-current');
        lnk.classList.add('link-body-emphasis');
      });
      this.classList.add('active');
      this.setAttribute('aria-current', 'page');
      this.classList.remove('link-body-emphasis');
    });
  });
});
