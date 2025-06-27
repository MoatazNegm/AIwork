/**
 * LAN Spines Operations JavaScript
 * Handles delete operations for LAN Spines table
 */

$(document).ready(function() {
  // Delete LAN Spine button click handler
  $(document).on("click", ".delete-lan-spine-btn", function(e) {
    e.stopPropagation();
    const index = $(this).data("index");
    
    Swal.fire({
      title: "Are you sure?",
      text: "This will delete this LAN spine configuration. This action cannot be undone!",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#3085d6",
      cancelButtonColor: "#d33",
      confirmButtonText: "Yes, delete it!"
    }).then((result) => {
      if (result.isConfirmed) {
        $.ajax({
          url: "/delete_lan_spine",
          type: "POST",
          data: {
            index: index
          },
          success: function(response) {
            if (response.status === "success") {
              toastr.success(response.message);
              updateLanSpinesTable(response.all_entries);
              refreshJsonDisplay();
            } else {
              toastr.error(response.message);
            }
          },
          error: function() {
            toastr.error("Failed to delete LAN spine");
          }
        });
      }
    });
  });
});
