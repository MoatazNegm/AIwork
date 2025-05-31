// JavaScript to handle compute node editing
$(function() {
  // Edit button click handler
  $(document).on('click', '.edit-compute-btn', function() {
    const index = $(this).data('index');
    if (index !== undefined) {
      loadComputeForEdit(index);
    }
  });

  // Function to load compute data for editing
  function loadComputeForEdit(index) {
    $.ajax({
      url: "/get_compute",
      type: "GET",
      data: { index: index },
      success: function(response) {
        if (response.status === "success") {
          // Get the compute entry
          const computeEntry = response.entry;
          const computeName = Object.keys(computeEntry)[0];
          const details = computeEntry[computeName];
          
          // Fill the form with compute data
          $("#computeName").val(computeName);
          $("#count").val(details.count);
          $("#wattage").val(details.wattage);
          $("#height").val(details.height);
          $("#weight").val(details.weight);
          
          // Clear all LAN inputs first
          for (let i = 1; i <= 6; i++) {
            $(`#lan_${i}_count`).val('');
            $(`#lan_${i}_speed`).val('');
          }
          
          // Fill LAN details if available
          for (let i = 1; i <= 6; i++) {
            const lanKey = `LAN_${i}`;
            if (details[lanKey]) {
              $(`#lan_${i}_count`).val(details[lanKey].count);
              $(`#lan_${i}_speed`).val(details[lanKey].speed);
            }
          }
          
          // Set edit index and update button text
          $("#computeEditIndex").val(index);
          $("#submitComputeBtn").text("Update Compute Node");
          $("#cancelComputeEditBtn").removeClass("d-none");
        } else {
          console.error("Error loading compute data:", response.message);
          toastr.error(response.message || "Error loading compute data");
        }
      },
      error: function() {
        console.error("Failed to load compute data");
        toastr.error("Failed to load compute data");
      }
    });
  }

  // Cancel edit button click
  $("#cancelComputeEditBtn").on('click', function() {
    // Reset form
    $("#computeForm")[0].reset();
    $("#computeEditIndex").val("");
    $("#submitComputeBtn").text("Add Compute Node");
    $(this).addClass("d-none");
  });
});

