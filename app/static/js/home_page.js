$(document).ready(function() {
    // Toggle visibility of the sidebar
    $('#toggle-sidebar').on('click', function() {
        const sidebar = $('#sidebar');
        sidebar.toggleClass('active');

        // Update the toggle button icon/text based on the sidebar state
        if (sidebar.hasClass('active')) {
            $(this).text("✖");
        } else {
            $(this).text("☰");
        }
    });

    // Close the sidebar via the "Close Sidebar" button inside the sidebar
    $('#close-sidebar').on('click', function() {
        const sidebar = $('#sidebar');
        sidebar.removeClass('active');
        $('#toggle-sidebar').text("☰");
    });

    // Show the alert modal if the form was submitted with empty fields
    var isEmpty = $('#isempty').val() === 'True';
    if (isEmpty) {
        $('#alertModal').modal('show');
    }

    // Prevent form submission with Enter and insert a new line instead
    $('#essay').on('keydown', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent form submission
            var start = this.selectionStart;
            var end = this.selectionEnd;
            var text = this.value;
            this.value = text.substring(0, start) + "\n" + text.substring(end);
            this.selectionStart = this.selectionEnd = start + 1;
        }
    });

    // Countdown Timer Logic
    let countdownSeconds;
    const countdownDisplay = document.getElementById("countdown-timer");
    let countdownInterval;

    function updateCountdown() {
        const minutes = Math.floor(countdownSeconds / 60);
        const seconds = countdownSeconds % 60;
        countdownDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        countdownSeconds--;
        if (countdownSeconds >= 0) {
            countdownInterval = setTimeout(updateCountdown, 1000);
        } else {
            countdownDisplay.textContent = "Time's up!";
        }
    }

    $('#start-countdown').on('click', function() {
        const time = countdownDisplay.textContent.trim();
        const timeParts = time.split(':');
        const minutesInput = parseInt(timeParts[0], 10) || 0;
        const secondsInput = parseInt(timeParts[1], 10) || 0;
        countdownSeconds = minutesInput * 60 + secondsInput;

        clearTimeout(countdownInterval); // Clear any existing countdown
        updateCountdown(); // Start the countdown
    });

    $('#stop-countdown').on('click', function() {
        clearTimeout(countdownInterval); // Stop the countdown timer
    });
});
