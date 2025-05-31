// Switches section row click and edit handler
$(function() {
  // Variables to track edit state
  let currentEditIndex = null;
  let hasUnsavedChanges = false;

  // Function to handle row clicks in the switches section
  function setupSwitchRowClickHandlers() {
    // Switches section row click
    $(document).on('click', '#switchesTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on the edit or delete button itself
      if ($(e.target).hasClass('edit-switch-btn') || $(e.target).hasClass('delete-switch-btn') || 
          $(e.target).closest('.edit-switch-btn').length || $(e.target).closest('.delete-switch-btn').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        loadSwitchForEdit(index);
      }
    });

    // Edit button click
    $(document).on('click', '.edit-switch-btn', function() {
      const index = $(this).data('index');
      if (index !== undefined) {
        loadSwitchForEdit(index);
      }
    });

    // Delete button click
    $(document).on('click', '.delete-switch-btn', function() {
      const index = $(this).data('index');
      const switchModel = $(this).data('model');
      
      if (index !== undefined && switchModel) {
        deleteSwitchAndLinks(index, switchModel);
      }
    });
  }

  // Function to load switch data for editing
  function loadSwitchForEdit(index) {
    // Always load the new switch data, regardless of unsaved changes
    $.ajax({
      url: "/get_switch",
      type: "GET",
      data: { index: index },
      success: function(response) {
        if (response.status === "success") {
          // Fill the form with switch data
          const switchData = response.entry;
          
          $("#switchModel").val(switchData.model);
          $("#switchPorts").val(switchData.ports);
          $("#switchSpeed").val(switchData.speed);
          $("#switchHeight").val(switchData.height);
          $("#switchWattage").val(switchData.wattage);
          $("#switchWeight").val(switchData.weight);
          
          // Optional fields
          if (switchData.uplink_count !== undefined) {
            $("#uplinkCount").val(switchData.uplink_count);
          } else {
            $("#uplinkCount").val("");
          }
          
          if (switchData.uplink_speed !== undefined) {
            $("#uplinkSpeed").val(switchData.uplink_speed);
          } else {
            $("#uplinkSpeed").val("");
          }
          
          // Set edit index and update button text
          $("#switchEditIndex").val(index);
          $("#submitSwitchBtn").text("Update Switch");
          $("#cancelSwitchEditBtn").removeClass("d-none");
          
          // Update edit state
          currentEditIndex = index;
          hasUnsavedChanges = false;
        } else {
          console.error("Error loading switch data:", response.message);
          toastr.error(response.message || "Error loading switch data");
        }
      },
      error: function() {
        console.error("Failed to load switch data");
        toastr.error("Failed to load switch data");
      }
    });
  }

  // Function to delete a switch and its links
  function deleteSwitchAndLinks(index, switchModel) {
    // Confirm deletion
    Swal.fire({
      title: 'Delete Switch',
      text: `Are you sure you want to delete the switch "${switchModel}"? This will also remove any references to this switch in LAN leaves and LAN spines.`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#d33',
      cancelButtonColor: '#3085d6',
      confirmButtonText: 'Yes, delete it!'
    }).then((result) => {
      if (result.isConfirmed) {
        // Send delete request
        $.ajax({
          url: "/delete_switch",
          type: "POST",
          data: { 
            index: index,
            model: switchModel
          },
          success: function(response) {
            if (response.status === "success") {
              toastr.success(response.message);
              
              // Reset form if we were editing this switch
              if (currentEditIndex === index) {
                $("#switchForm")[0].reset();
                $("#switchEditIndex").val("");
                $("#submitSwitchBtn").text("Add Switch");
                $("#cancelSwitchEditBtn").addClass("d-none");
                currentEditIndex = null;
                hasUnsavedChanges = false;
              }
              
              // Update switches table
              updateSwitchesTable(response.all_entries);
              
              // Update switch dropdowns in other forms
              updateSwitchDropdowns(response.all_entries);
              
              // Refresh JSON display
              refreshJsonDisplay();
            } else {
              toastr.error(response.message);
            }
          },
          error: function() {
            toastr.error("Failed to delete switch");
          }
        });
      }
    });
  }

  // Cancel edit button click
  $("#cancelSwitchEditBtn").on('click', function() {
    // Reset form
    $("#switchForm")[0].reset();
    $("#switchEditIndex").val("");
    $("#submitSwitchBtn").text("Add Switch");
    $(this).addClass("d-none");
    
    // Reset edit state
    currentEditIndex = null;
    hasUnsavedChanges = false;
  });

  // Form change detection
  $("#switchForm").on('input change', function() {
    hasUnsavedChanges = true;
  });

  // Form submission
  $("#switchForm").on('submit', function(e) {
    e.preventDefault();
    
    $.ajax({
      url: "/submit_switch",
      type: "POST",
      data: $(this).serialize(),
      success: function(response) {
        if (response.status === "success") {
          toastr.success(response.message);
          
          // Reset form and edit state
          $("#switchForm")[0].reset();
          $("#switchEditIndex").val("");
          $("#submitSwitchBtn").text("Add Switch");
          $("#cancelSwitchEditBtn").addClass("d-none");
          currentEditIndex = null;
          hasUnsavedChanges = false;
          
          // Update switches table
          updateSwitchesTable(response.all_entries);
          
          // Update switch dropdowns in other forms
          updateSwitchDropdowns(response.all_entries);
          
          // Refresh JSON display
          refreshJsonDisplay();
        } else {
          toastr.error(response.message);
        }
      },
      error: function() {
        toastr.error("Failed to submit switch");
      }
    });
  });

  // Add delete button to switch rows
  function addDeleteButtonToSwitchRows() {
    $('#switchesTableBody tr').each(function() {
      const row = $(this);
      const index = row.data('index');
      const model = row.find('td:first').text();
      const actionsCell = row.find('td:last');
      
      // Check if delete button already exists
      if (actionsCell.find('.delete-switch-btn').length === 0) {
        // Add delete button
        actionsCell.append(
          `<button class="btn btn-sm btn-danger delete-switch-btn ml-2" data-index="${index}" data-model="${model}">
            <i class="fas fa-trash"></i> Delete
          </button>`
        );
      }
    });
  }

  // Initialize handlers and add delete buttons
  setupSwitchRowClickHandlers();
  addDeleteButtonToSwitchRows();
  
  // Re-add delete buttons after any AJAX updates
  $(document).ajaxComplete(function() {
    setTimeout(addDeleteButtonToSwitchRows, 100); // Small delay to ensure DOM is updated
  });
});
