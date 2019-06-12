import Translater from 'translater.js'
import Siema from 'siema';

const main = () => {
  // A convenient i18n solution
  var tran = new Translater()
  tran.setLang('en')
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

  // Detect browser language and set automatically
  switch(getFirstBrowserLanguage()) {
    case "zh-CN":
      tran.setLang('zh')
      break
    case "fr-FR":
      tran.setLang('fr')
      break
    default:
      tran.setLang('en')
  }

  // Carousel Gallery Initiate
  let statsSiema = new Siema({
    selector: '.siema',
    duration: 800,
    easing: 'ease-in-out',
    threshold: 20,
    loop: false,
    rtl: false,
    onInit: () => {},
    onChange: () => {},
  });
  document.querySelector('.arrow-strip-prev').addEventListener('click', () => {
    statsSiema.prev()
  })
  document.querySelector('.arrow-strip-next').addEventListener('click', () => {
    statsSiema.next()
  })

  // Dealing with input and output
  document.querySelector('.cl-submit').addEventListener('click', () => {
    let fdat = new FormData()
    fdat.append('city', document.querySelector('.cl-input').value)
    fetch('http://pn-i.club/api/city/predict', {
        method: 'POST',
        body: fdat
      })
      .then(response => response.json())
      .then(response => {
        console.log(console.log(document.getElementById("in").value))
        console.log(response)
        setReport(response.area)
      })
      .catch(error => console.error('Placename Submit Error.', error))
  })
}

function getFirstBrowserLanguage() {
  var nav = window.navigator,
    browserLanguagePropertyKeys = ['language', 'browserLanguage', 'systemLanguage', 'userLanguage'],
    i,
    language,
    len,
    shortLanguage = null;

  // support for HTML 5.1 "navigator.languages"
  if (Array.isArray(nav.languages)) {
    for (i = 0; i < nav.languages.length; i++) {
      language = nav.languages[i];
      len = language.length;
      if (!shortLanguage && len) {
        shortLanguage = language;
      }
      if (language && len > 2) {
        return language;
      }
    }
  }

  // support for other well known properties in browsers
  for (i = 0; i < browserLanguagePropertyKeys.length; i++) {
    language = nav[browserLanguagePropertyKeys[i]];
    len = language.length;
    if (!shortLanguage && len) {
      shortLanguage = language;
    }
    if (language && len > 2) {
      return language;
    }
  }

  return shortLanguage;
}

function setReport(data) {
  let dict = {
    EastAsia: {
      title: "East Asia",
      description: "The writing system in East Asia is usually ideographic, which generates clear syllable boundaries in its place names. Some common spelling patterns, including “-eng”, “-ang” can be seen."
    },
    "S&SEAsia": {
      title: "South & Southeast Asia",
      description: 'The features in Southeast Asia and South Asia placenames is somewhat similar to that of East Asia. However, the average word length is dramatically shorter because each ideograph represents one word. The “Syllable boundary” in East Asia placename words simply becomes whitespaces here.'
    },
    SSAfrica: {
      title: "Non-Arabic Africa",
      description: 'Place names in Africa (excluding the northern Arabic cultural region) are diverse in characteristics, making them harder to recognize. We can occasionally notice a Europe impact, especially France, on their place names.'
    },
    Oceania: {
      title: "Oceania Islands",
      description: 'Place names in Oceania are diverse in characteristics and meanwhile short in collected data. This makes them nearly impossible to be classified.'
    },
    WEurope: {
      title: "Western Europe",
      description: 'Place names in West Europe typically demonstrate a low vowel-consonant ratio, which means the pronunciation sounds “harder”. Some common suffixes like “-burg”, “-eaux” can be observed.'
    },
    EEurope: {
      title: "Eastern Europe",
      description: 'Place names in East Europe are usually constructed by a single long word, which makes them relatively easier to be recognized. Some Slavic suffixes like “-sk” are commonly seen here.'
    },
    EnUsAuNz: {
      title: "English Spoken Region",
      description: 'Place names from English-spoken countries (North America, Australia, New Zealand, and the UK) show features in the English language.'
    },
    Latinos: {
      title: "Latin Region",
      description: "Place names in Latin Region, including Latin America, Spain, and Portugal, are made up of relatively short words. Some common prefixes like “de”, “le”, “san” can be observed."
    },
    Arabics: {
      title: "Arabic Region",
      description: 'Place names in the Arabic Cultural Region usually suggest some unique patterns, including "Al", "Ah", and "-j". Some difficultly pronounced combinations of letters are detected.'
    }
  }
  let elToSet = document.querySelectorAll('.single-region')
  for (let i = 0; i < 3; i++) {
    let region = data[i][1]
    elToSet[i].querySelector('h3').innerText = dict[region].title
    elToSet[i].querySelector('.single-region-cost').innerText = data[i][0]
    elToSet[i].querySelector('.single-region-bar').style.width = ((6 + data[i][0]) * 30) + "px"
    elToSet[i].querySelector('p').innerText = dict[region].description
  }
}

function fadeOut(el) {
  el.style.opacity = 1;

  (function fade() {
    if ((el.style.opacity -= .1) < 0) {
      el.style.display = "none";
    } else {
      requestAnimationFrame(fade);
    }
  })();
};

function fadeIn(el, display) {
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