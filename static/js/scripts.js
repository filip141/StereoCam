function buildTable(objectList){
    // Build table
    var $table = $('#objects');
    // Clear table
    $table.html('');
    for (var i = 0 ; i < objectList.length; i++){
        var $tr = $('<tr>').appendTo($table);
        console.log($tr)
        for (var j = 0 ; j < objectList[i].length; j++){
            $tr.append('<td>' + objectList[i][j] + '</td>');
        }
        $table.append('</tr>');
    }
}

function updateTable() {
    $.ajax({
        url: '/get_objects',
        method: 'GET',
        dataType: "json",
        success: function(data) {
            buildTable(data.objects);
        },
         error: function(error) {
            console.log(error);
        }
    });
}
setInterval(updateTable, 1000);
