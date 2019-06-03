import Translater from 'translater.js'

const main = () => {
    // A convenient i18n solution
    var tran = new Translater()
    document.querySelectorAll('.trans').forEach((el) => {
        el.addEventListener('click', () => {
            let dimPanel = document.querySelector('.dim-panel');
            fadeIn(dimPanel)
            setTimeout(() => {
                tran.setLang(el.innerHTML)
                fadeOut(dimPanel)
            }, 500)
        })
    })

}

function fadeOut(el){
    el.style.opacity = 1;
  
    (function fade() {
      if ((el.style.opacity -= .1) < 0) {
        el.style.display = "none";
      } else {
        requestAnimationFrame(fade);
      }
    })();
  };
  
  function fadeIn(el, display){
    el.style.opacity = 0;
    el.style.display = display || "block";
  
    (function fade() {
      var val = parseFloat(el.style.opacity);
      if (!((val += .1) > 1)) {
        el.style.opacity = val;
        requestAnimationFrame(fade);
      }
    })();
  };

document.addEventListener('DOMContentLoaded', main)
