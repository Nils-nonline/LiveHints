let menu = document.querySelector(".fixed-menu");

menu.addEventListener('click', function(event) {
    let clickTarget = event.target;
    let activeBtn = document.querySelector('.active');

    if (clickTarget.classList.contains('nav-link')) {
        if (activeBtn) {
            activeBtn.classList.remove('active');
        }
        clickTarget.classList.add('active');
    }
});

let classLink = '.main-link';

window.onscroll = function () {
    let h = document.documentElement.clientHeight;

    if (window.scrollY >= h) {
        classLink = '.info-link';
    } else {
        classLink = '.main-link';
    }

    let activeBtn = document.querySelector('.active');
    let newActiveBtn = document.querySelector(classLink);

    if (newActiveBtn && !newActiveBtn.classList.contains('active')) {
        newActiveBtn.classList.add('active');
        if (activeBtn) {
            activeBtn.classList.remove('active');
        }
    }
};



