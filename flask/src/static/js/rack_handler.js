// JavaScript to handle rack row editing, deletion, and search
$(function() {
  // Edit button click handler
  $(document).on('click', '.edit-rack-row-btn', function() {
    const index = $(this).data('index');
    if (index !== undefined) {
      loadRackRowForEdit(index);
    }
  });

  // Add click handler for clickable rows in rack rows table
  $(document).on('click', '#rackRowsTableBody tr.clickable-row', function(e) {
    // Don't trigger if clicking on a button inside the row
    if (!$(e.target).closest('button').length) {
      const index = $(this).data('index');
      if (index !== undefined) {
        loadRackRowForEdit(index);
      }
    }
  });

  // Delete button click handler
  $(document).on('click', '.delete-rack-row-btn', function() {
    const index = $(this).data('index');
    if (index !== undefined) {
      deleteRackRow(index);
    }
  });

  // Search input handler
  $("#rackRowSearchInput").on("keyup", function() {
    const searchText = $(this).val().toLowerCase();
    $("#rackRowsTableBody tr").filter(function() {
      $(this).toggle($(this).text().toLowerCase().indexOf(searchText) > -1);
    });
  });

  // Function to load rack row data for editing
  function loadRackRowForEdit(index) {
    $.ajax({
      url: "/get_rack_row",
      type: "GET",
      data: { index: index },
      success: function(response) {
        if (response.status === "success") {
          // Get the rack row entry
          const rackRowEntry = response.entry;
          const groupName = Object.keys(rackRowEntry)[0];
          const details = rackRowEntry[groupName];
          
          // Fill the form with rack row data
          $("#groupName").val(groupName);
          $("#racks").val(details.Racks);
          $("#groupCount").val(details.group_count);
          $("#rackToRack").val(details.rack_to_rack);
          $("#rowToNextRow").val(details.Row_to_next_row);
          
          // Set edit index and update button text
          $("#rackRowEditIndex").val(index);
          $("#submitRackRowBtn").text("Update Rack Row");
          $("#cancelRackRowEditBtn").removeClass("d-none");
        } else {
          console.error("Error loading rack row data:", response.message);
          toastr.error(response.message || "Error loading rack row data");
        }
      },
      error: function() {
        console.error("Failed to load rack row data");
        toastr.error("Failed to load rack row data");
      }
    });
  }

  // Function to delete rack row
  function deleteRackRow(index) {
    // Confirm deletion
    Swal.fire({
      title: "Are you sure?",
      text: "You won't be able to revert this!",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#3085d6",
      cancelButtonColor: "#d33",
      confirmButtonText: "Yes, delete it!"
    }).then((result) => {
      if (result.isConfirmed) {
        $.ajax({
          url: "/delete_rack_row",
          type: "POST",
          data: { index: index },
          success: function(response) {
            if (response.status === "success") {
              toastr.success(response.message);
              
              // Update rack rows table
              updateRackRowsTable(response.all_entries);
              
              // Refresh JSON display
              refreshJsonDisplay();
            } else {
              toastr.error(response.message);
            }
          },
          error: function() {
            toastr.error("Failed to delete rack row");
          }
        });
      }
    });
  }

  // Cancel edit button click
  $("#cancelRackRowEditBtn").on('click', function() {
    // Reset form
    $("#rackRowForm")[0].reset();
    $("#rackRowEditIndex").val("");
    $("#submitRackRowBtn").text("Add Rack Row");
    $(this).addClass("d-none");
  });
});
