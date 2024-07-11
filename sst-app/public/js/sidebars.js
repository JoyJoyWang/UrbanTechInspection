/* global bootstrap: false */
(() => {
  'use strict'

  const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  tooltipTriggerList.forEach(tooltipTriggerEl => {
    new bootstrap.Tooltip(tooltipTriggerEl)
  })

  // Add event listeners for sidebar links
  document.querySelectorAll('.nav-link').forEach(function(link) {
    link.addEventListener('click', function() {
      // Remove 'active' class from all links
      document.querySelectorAll('.nav-link').forEach(function(navLink) {
        navLink.classList.remove('active');
      });
      // Add 'active' class to the clicked link
      link.classList.add('active');
      
      // Hide all sections
      document.getElementById('runModelsContent').style.display = 'none';
      document.getElementById('resultViewerContent').style.display = 'none';
      // Show the clicked section
      if (link.id === 'runModelsLink') {
        document.getElementById('runModelsContent').style.display = 'block';
      } else if (link.id === 'previewLink') {
        document.getElementById('resultViewerContent').style.display = 'block';
      }
      // Add additional conditions for other sections if needed
    });
  });

  // Sign out event listener
  document.getElementById('signOut').addEventListener('click', function() {
    window.location.href = 'signIn.html';
  });


  // Change steps
  document.getElementById('btnradio1').addEventListener('change', function() {
    if (this.checked) {
      document.getElementById('step1-card').style.display = 'block';
      document.getElementById('step2-card').style.display = 'none';
    }
  });

  document.getElementById('btnradio2').addEventListener('change', function() {
    if (this.checked) {
      document.getElementById('step1-card').style.display = 'none';
      document.getElementById('step2-card').style.display = 'block';
    }
  });



})();