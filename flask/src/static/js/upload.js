// Add this to the existing JavaScript in index.html

// File Upload Functionality
$(function() {
  // Add Upload Config button to navbar
  const uploadBtn = $('<li class="nav-item"><button type="button" class="btn btn-primary btn-sm" id="uploadConfigBtn"><i class="fas fa-upload"></i> Upload Config</button></li>');
  $('.navbar-nav.ml-auto').prepend(uploadBtn);
  
  // Handle Upload Config button click
  $('#uploadConfigBtn').on('click', function() {
    $('#uploadFileModal').modal('show');
  });
  
  // Handle file upload form submission
  $('#submitUploadBtn').on('click', function() {
    const fileInput = $('#configFile')[0];
    if (fileInput.files.length === 0) {
      toastr.error('Please select a file to upload');
      return;
    }
    
    const formData = new FormData($('#uploadFileForm')[0]);
    
    $.ajax({
      url: '/upload_config_file',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      success: function(response) {
        $('#uploadFileModal').modal('hide');
        
        if (response.status === 'success') {
          // Display results
          let resultHtml = '<div class="alert alert-success">File uploaded and processed successfully!</div>';
          resultHtml += '<h5>Summary:</h5>';
          resultHtml += '<ul>';
          
          for (const [category, counts] of Object.entries(response.result)) {
            const displayCategory = category.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
            resultHtml += `<li>${displayCategory}: ${counts.added} added, ${counts.updated} updated</li>`;
          }
          
          resultHtml += '</ul>';
          
          $('#uploadResultContent').html(resultHtml);
          $('#uploadResultModal').modal('show');
          
          // Refresh the page after a short delay to show updated data
          setTimeout(function() {
            location.reload();
          }, 3000);
        } else {
          toastr.error(response.message || 'Error processing file');
        }
      },
      error: function() {
        $('#uploadFileModal').modal('hide');
        toastr.error('Error uploading file');
      }
    });
  });
});
