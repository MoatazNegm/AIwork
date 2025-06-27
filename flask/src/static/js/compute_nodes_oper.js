/**
 * Compute Nodes Operations JavaScript
 * Handles delete operations for Compute Nodes table
 */

$(document).ready(function() {
  // Delete Compute Node button click handler
  $(document).on("click", ".delete-compute-btn", function(e) {
    e.stopPropagation();
    const index = $(this).data("index");
    const computeName = $(this).data("compute-name");
    
    Swal.fire({
      title: "Are you sure?",
      text: "This will delete this compute node configuration. This action cannot be undone!",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#3085d6",
      cancelButtonColor: "#d33",
      confirmButtonText: "Yes, delete it!"
    }).then((result) => {
      if (result.isConfirmed) {
        $.ajax({
          url: "/delete_compute",
          type: "POST",
          data: {
            index: index,
            compute_name: computeName
          },
          success: function(response) {
            if (response.status === "success") {
              toastr.success(response.message);
              updateComputeTable(response.all_entries);
              refreshJsonDisplay();
            } else {
              toastr.error(response.message);
            }
          },
          error: function() {
            toastr.error("Failed to delete compute node");
          }
        });
      }
    });
  });
});
