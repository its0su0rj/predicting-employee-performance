// =========================
// MOBILE NAVIGATION
// =========================

const menuBtn = document.querySelector(".menu-btn");
const navLinks = document.querySelector(".nav-links");

menuBtn.addEventListener("click", () => {
  navLinks.classList.toggle("active");
});

// Close menu after click (mobile)

document.querySelectorAll(".nav-links a").forEach(link => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("active");
  });
});


// =========================
// SCROLL REVEAL ANIMATION
// =========================

const reveals = document.querySelectorAll(".reveal");

const revealOnScroll = () => {

  reveals.forEach((el) => {

    const windowHeight = window.innerHeight;
    const top = el.getBoundingClientRect().top;

    if (top < windowHeight - 80) {
      el.classList.add("active");
    }

  });

};

window.addEventListener("scroll", revealOnScroll);
window.addEventListener("load", revealOnScroll);


// =========================
// NAVBAR BACKGROUND ON SCROLL
// =========================

const header = document.querySelector(".header");

window.addEventListener("scroll", () => {

  if (window.scrollY > 20) {

    header.style.background =
      "rgba(8,12,22,.55)";

    header.style.borderBottom =
      "1px solid rgba(255,255,255,.06)";

    header.style.boxShadow =
      "0 10px 35px rgba(0,0,0,.22)";

  } else {

    header.style.background = "transparent";
    header.style.borderBottom = "none";
    header.style.boxShadow = "none";

  }

});


// =========================
// LIGHT PARALLAX EFFECT
// =========================

const blobs = document.querySelectorAll(".blob");

window.addEventListener("mousemove", (e) => {

  const x = e.clientX / window.innerWidth;
  const y = e.clientY / window.innerHeight;

  blobs.forEach((blob, index) => {

    const speed = (index + 1) * 12;

    blob.style.transform =
      `translate(
        ${x * speed}px,
        ${y * speed}px
      )`;

  });

});


// =========================
// BUTTON MICRO INTERACTION
// =========================

const buttons = document.querySelectorAll(".btn");

buttons.forEach((btn) => {

  btn.addEventListener("mouseenter", () => {
    btn.style.transform = "translateY(-3px) scale(1.01)";
  });

  btn.addEventListener("mouseleave", () => {
    btn.style.transform = "";
  });

});


// =========================
// SECTION ACTIVE NAV LINK
// =========================

const sections = document.querySelectorAll("section");
const navItems = document.querySelectorAll(".nav-links a");

window.addEventListener("scroll", () => {

  let current = "";

  sections.forEach((section) => {

    const sectionTop = section.offsetTop - 120;

    if (scrollY >= sectionTop) {
      current = section.getAttribute("id");
    }

  });

  navItems.forEach((link) => {

    link.classList.remove("active-link");

    if (link.getAttribute("href") === "#" + current) {
      link.classList.add("active-link");
    }

  });

});


// =========================
// PERFORMANCE FRIENDLY LOAD
// =========================

window.addEventListener("load", () => {

  document.body.style.opacity = "1";

});
