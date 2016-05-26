//JQuerry events
String.prototype.isNumber = function(){return /^\d+$/.test(this);}
String.prototype.isFloatNumber = function(){return /^[0-9.]+$/.test(this);}

function checkParameters(stringOne, stringTwo, integerParam, floatParam){
    var condition = true;
    if(stringOne === ""){
        $('.warning-dev1').html("")
        addWarning($('.warning-dev1'), "String should not be empty");
    }
    condition = condition & (stringOne !== "");
    if(stringTwo === ""){
        $('.warning-dev2').html("")
        addWarning($('.warning-dev2'), "String should not be empty");
    }
    condition = condition & (stringTwo !== "");
    if(!integerParam.isNumber()){
        $('.warning-Nobj').html("")
        addWarning($('.warning-Nobj'), "N Objects should be integer");
    }
    condition = condition & integerParam.isNumber();
    if(!floatParam.isFloatNumber()){
        $('.warning-HParam').html("")
        addWarning($('.warning-HParam'), "N Objects should be integer");
    }
    condition = condition & floatParam.isFloatNumber();
    return condition;
}


$(document).ready(function(){
    $('#camchange').click(function(event){
        // Camera devices
        var devOne = $('#inputDev1').val();
        var devTwo = $('#inputDev2').val();
        // Program parameters
        var nObj = $('#inputNobj').val();
        var horizont = $('#inputHParam').val();
        var validator = checkParameters(devOne, devTwo, nObj, horizont);
        if(validator)
        {
            var data = {
                'devOne': devOne,
                'devTwo': devTwo,
                'nObj': nObj,
                'horizont': horizont,
            }
            $.ajax({
                url: '/change_camera',
                method: 'POST',
                dataType: "json",
                data: JSON.stringify(data, null, '\t'),
                contentType: 'application/json;charset=UTF-8',
                success: function(data) {
                    location.reload();
                    alert(data.status);
                    location.reload();
                },
                 error: function(error) {
                    console.log(error);
                }
            });
        }
        event.preventDefault();
    });

});


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

function addWarning(object, text){

      object.append("<span class='red-warning' style='color:red;font-weight:normal;'><i>  " + text + "</i></span>");
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
