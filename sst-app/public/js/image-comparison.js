const sliders = document.querySelectorAll(".image-comparison .slider");

sliders.forEach(slider => {
  slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";
    const imageComparison = e.target.closest(".image-comparison");
    const beforeImage = imageComparison.querySelector(".before-image");
    const sliderLine = imageComparison.querySelector(".slider-line");
    const sliderIcon = imageComparison.querySelector(".slider-icon");

    beforeImage.style.width = sliderValue;
    sliderLine.style.left = sliderValue;
    sliderIcon.style.left = sliderValue;
  });
});
