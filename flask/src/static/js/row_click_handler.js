// Row Click Handler for all sections
$(function() {
  // Variables to track edit state
  let currentEditSection = null;
  let currentEditIndex = null;
  let hasUnsavedChanges = false;

  // Function to handle row clicks for all sections
  function setupRowClickHandlers() {
    // Switches section
    $(document).on('click', '#switchesTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on the edit button itself
      if ($(e.target).hasClass('edit-switch-btn') || $(e.target).closest('.edit-switch-btn').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('switches', index);
      } else {
        console.error('Missing data-index attribute on switch row');
      }
    });

    // LAN Leaves section
    $(document).on('click', '#lanLeavesTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('lan_leaves', index);
      } else {
        console.error('Missing data-index attribute on LAN leaf row');
      }
    });

    // LAN Spines section
    $(document).on('click', '#lanSpinesTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('lan_spines', index);
      } else {
        console.error('Missing data-index attribute on LAN spine row');
      }
    });

    // Compute Nodes section
    $(document).on('click', '#computeTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('compute', index);
      } else {
        console.error('Missing data-index attribute on compute row');
      }
    });

    // Storage Blocks section
    $(document).on('click', '#storageTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('storage', index);
      } else {
        console.error('Missing data-index attribute on storage row');
      }
    });

    // Cables section
    $(document).on('click', '#cablesTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('cables', index);
      } else {
        console.error('Missing data-index attribute on cable row');
      }
    });

    // Rack Rows section
    $(document).on('click', '#rackRowsTableBody tr.clickable-row', function(e) {
      // Don't trigger if clicking on buttons
      if ($(e.target).is('button') || $(e.target).closest('button').length) {
        return;
      }
      
      const index = $(this).data('index');
      if (index !== undefined) {
        handleSectionChange('rack_rows', index);
      } else {
        console.error('Missing data-index attribute on rack row');
      }
    });
  }

  // Function to handle section changes and cancel unconfirmed edits
  function handleSectionChange(newSection, newIndex) {
    // Check if we're already editing something
    if (hasUnsavedChanges && (currentEditSection !== newSection || currentEditIndex !== newIndex)) {
      // Cancel the current edit
      cancelEdit(currentEditSection);
    }
    
    // Trigger edit for the new row
    triggerEdit(newSection, newIndex);
    
    // Update current edit tracking
    currentEditSection = newSection;
    currentEditIndex = newIndex;
    hasUnsavedChanges = true;
  }

  // Function to trigger edit for a specific section and index
  function triggerEdit(section, index) {
    // First check if the edit button exists
    let editButton;
    
    switch(section) {
      case 'switches':
        editButton = $('.edit-switch-btn[data-index="' + index + '"]');
        break;
      case 'lan_leaves':
        editButton = $('.edit-lan-leaf-btn[data-index="' + index + '"]');
        break;
      case 'lan_spines':
        editButton = $('.edit-lan-spine-btn[data-index="' + index + '"]');
        break;
      case 'compute':
        editButton = $('.edit-compute-btn[data-index="' + index + '"]');
        break;
      case 'storage':
        editButton = $('.edit-storage-btn[data-index="' + index + '"]');
        break;
      case 'cables':
        editButton = $('.edit-cable-btn[data-index="' + index + '"]');
        break;
      case 'rack_rows':
        editButton = $('.edit-rack-row-btn[data-index="' + index + '"]');
        break;
    }
    
    // Only click if the button exists
    if (editButton && editButton.length > 0) {
      editButton.click();
    } else {
      console.error('Edit button not found for ' + section + ' with index ' + index);
      // Reset edit state since we couldn't trigger the edit
      currentEditSection = null;
      currentEditIndex = null;
      hasUnsavedChanges = false;
    }
  }

  // Function to cancel edit for a specific section
  function cancelEdit(section) {
    let cancelButton;
    
    switch(section) {
      case 'switches':
        cancelButton = $('#cancelSwitchEditBtn');
        break;
      case 'lan_leaves':
        cancelButton = $('#cancelLanLeafEditBtn');
        break;
      case 'lan_spines':
        cancelButton = $('#cancelLanSpineEditBtn');
        break;
      case 'compute':
        cancelButton = $('#cancelComputeEditBtn');
        break;
      case 'storage':
        cancelButton = $('#cancelStorageEditBtn');
        break;
      case 'cables':
        cancelButton = $('#cancelCableEditBtn');
        break;
      case 'rack_rows':
        cancelButton = $('#cancelRackRowEditBtn');
        break;
    }
    
    // Only click if the button exists
    if (cancelButton && cancelButton.length > 0) {
      cancelButton.click();
    }
    
    // Reset edit state
    hasUnsavedChanges = false;
  }

  // Handle tab changes to cancel edits when switching sections
  $('a[data-toggle="tab"]').on('show.bs.tab', function(e) {
    if (hasUnsavedChanges) {
      cancelEdit(currentEditSection);
      currentEditSection = null;
      currentEditIndex = null;
      hasUnsavedChanges = false;
    }
  });

  // Reset edit state when a form is submitted successfully
  $(document).on('formSubmitSuccess', function() {
    currentEditSection = null;
    currentEditIndex = null;
    hasUnsavedChanges = false;
  });

  // Setup form change detection
  function setupFormChangeDetection() {
    // For each form, detect changes to mark as unsaved
    $('#switchForm, #lanLeafForm, #lanSpineForm, #computeForm, #storageForm, #cableForm, #rackRowForm').on('input change', function() {
      hasUnsavedChanges = true;
    });
    
    // When cancel buttons are clicked, reset the edit state
    $('#cancelSwitchEditBtn, #cancelLanLeafEditBtn, #cancelLanSpineEditBtn, #cancelComputeEditBtn, #cancelStorageEditBtn, #cancelCableEditBtn, #cancelRackRowEditBtn').on('click', function() {
      currentEditSection = null;
      currentEditIndex = null;
      hasUnsavedChanges = false;
    });
  }

  // Add clickable-row class and data-index to rows that don't have them
  function fixTableRows() {
    // Function to update table rows with proper classes and attributes
    function updateTableRows(tableId, editBtnClass) {
      const tableBody = $('#' + tableId);
      if (tableBody.length === 0) return;
      
      tableBody.find('tr').each(function(index) {
        // Add clickable-row class if missing
        if (!$(this).hasClass('clickable-row')) {
          $(this).addClass('clickable-row');
        }
        
        // Add data-index if missing
        if ($(this).data('index') === undefined) {
          $(this).attr('data-index', index);
        }
        
        // Check if edit button exists, add if missing
        const rowIndex = $(this).data('index');
        const editBtn = $(this).find('.' + editBtnClass);
        
        if (editBtn.length === 0) {
          // Add edit button to last cell
          const lastCell = $(this).find('td:last-child');
          if (lastCell.length > 0) {
            lastCell.append('<button class="btn btn-sm btn-info ' + editBtnClass + '" data-index="' + rowIndex + '">Edit</button>');
          } else if ($(this).find('td').length > 0) {
            // If no last cell found but row has cells, add a new cell with edit button
            $(this).append('<td><button class="btn btn-sm btn-info ' + editBtnClass + '" data-index="' + rowIndex + '">Edit</button></td>');
          }
        } else {
          // Ensure edit button has correct data-index
          editBtn.attr('data-index', rowIndex);
        }
      });
    }
    
    // Update all tables
    updateTableRows('switchesTableBody', 'edit-switch-btn');
    updateTableRows('lanLeavesTableBody', 'edit-lan-leaf-btn');
    updateTableRows('lanSpinesTableBody', 'edit-lan-spine-btn');
    updateTableRows('computeTableBody', 'edit-compute-btn');
    updateTableRows('storageTableBody', 'edit-storage-btn');
    updateTableRows('cablesTableBody', 'edit-cable-btn');
    updateTableRows('rackRowsTableBody', 'edit-rack-row-btn');
  }

  // Initialize handlers and fix table rows
  fixTableRows();
  setupRowClickHandlers();
  setupFormChangeDetection();
  
  // Re-fix table rows after any AJAX updates
  $(document).ajaxComplete(function() {
    setTimeout(fixTableRows, 100); // Small delay to ensure DOM is updated
  });
});
