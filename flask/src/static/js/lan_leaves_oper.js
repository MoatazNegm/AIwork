/**
 * LAN Leaves Operations JavaScript
 * Handles delete operations for LAN Leaves table
 */

$(document).ready(function() {
  // Delete LAN Leaf button click handler
  $(document).on("click", ".delete-lan-leaf-btn", function(e) {
    e.stopPropagation();
    const index = $(this).data("index");
    
    Swal.fire({
      title: "Are you sure?",
      text: "This will delete this LAN leaf configuration. This action cannot be undone!",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#3085d6",
      cancelButtonColor: "#d33",
      confirmButtonText: "Yes, delete it!"
    }).then((result) => {
      if (result.isConfirmed) {
        $.ajax({
          url: "/delete_lan_leaf",
          type: "POST",
          data: {
            index: index
          },
          success: function(response) {
            if (response.status === "success") {
              toastr.success(response.message);
              updateLanLeavesTable(response.all_entries);
              refreshJsonDisplay();
            } else {
              toastr.error(response.message);
            }
          },
          error: function() {
            toastr.error("Failed to delete LAN leaf");
          }
        });
      }
    });
  });
});
