var axiosInstance = axios.create({
  baseURL: "https://gateway.watsonplatform.net/natural-language-classifier/api/v1/classifiers",
  timeout: 10000,
  headers: {
    Authorization: "Basic " + btoa("31419ee2-b7cf-4eba-9836-a4eba9c45081" + ":" + "hddFaFXldRRA")
  }
});

var database = firebase.database();
var selected = {
    classifier_id: null,
    classifier_name: null
}

function save(classifier) {
  database.ref('classifiers/' + classifier.classifier_id).set(classifier);
}

function retrieveClassifiers() {
    database.ref("/classifiers").on("value", snapshot => {
        $("#classifier_list").html("");
        if (snapshot.val() !== null) {
            classifiers = Object.values(snapshot.val());
            classifiers.forEach(classifier => {
                addNewClassifier(classifier);
            });
        }
     });
}

function selectRow(event, tr) {
    classifier_id = $(tr).find(".classifier_id").html();
    checkAndUpdateClassifier(classifier_id);
}

function selectClassifier(classifier) {
    selected.classifier_id = classifier.classifier_id;
    selected.classifier_name = classifier.name;
    $("#classifier_id").html(selected.classifier_id);
    $("#classifier_name").html(selected.classifier_name);
}

function addNewClassifier(classifier) {
    classColor = classifier.status == "Training" ? "warning" : "success";
    $("#classifier_list").append('<tr onclick="selectRow(event, this)" id="classifier_' + classifier.classifier_id + '" class="' + classColor + '">'+
       '<td class="classifier_id">' + classifier.classifier_id + '</td>' +
       '<td class="classifier_name">' + classifier.name + '</td>' +
       '<td>' + classifier.created + '</td>' +
       '<td class="status">' + classifier.status + '</td>' +
       '<td>' +
         '<button type="button" onClick="deleteClassifier(\'' + classifier.classifier_id + '\')" class="btn btn-danger btn-sm">' +
           '<span class="glyphicon glyphicon-remove"></span> Delete ' +
         '</button>' +
       '</td>' +
	 '</tr>');
	 save(classifier);
	 checkAndUpdateClassifier(classifier.classifier_id);
}

function checkAndUpdateClassifier(classifier_id) {
    console.log("Atualizando status: " + classifier_id);
    axiosInstance.get("/" + classifier_id)
     .then(res => {
        var classifier = res.data;
        if (classifier.status == "Training") {
            setTimeout(function(){checkAndUpdateClassifier(classifier_id)}, 3000)
        } else {
            $("#classifier_" + classifier_id + " > td.status").html(classifier.status);
            $("#classifier_" + classifier_id).attr("class", "success");
            save(classifier);
            selectClassifier(classifier);
        }
     })
    .catch(err => {
        console.error(err)
    });
}

function uploadDataset() {
  var data = new FormData();
  data.append('training_data', document.getElementById('trainingFIle').files[0]);
  data.append('training_metadata', JSON.stringify({
    language: "en",
    name: $("#classifierName").val()
  }));
  var config = {
    onUploadProgress: function(progressEvent) {
      var percentCompleted = Math.round( (progressEvent.loaded * 100) / progressEvent.total );
      $("#progress_upload").css("width", percentCompleted + "%")
    }
  };
  axiosInstance.post('/', data, config)
    .then(function (res) {
      console.log(res.data);
      addNewClassifier(res.data);
    })
    .catch(function (err) {
      console.error(err)
    });

   return false;
}

function processClassify() {
    classify($("#text_to_classify").val());
    return false;
}

function classify(text) {
  axiosInstance.get("/" + selected.classifier_id + "/classify?text=" + encodeURI(text))
    .then(function (res) {
      classifier = res.data;
      $("#sentences_list").prepend(
        '<tr>' +
          '<td>' + classifier.classifier_id + '</td>' +
          '<td>' + text + '</td>' +
          '<td>' + classifier.classes[0].confidence + '</td>' +
          '<td>' + classifier.classes[1].confidence + '</td>' +
        '</tr>'
      )
    })
    .catch(function (err) {
      alert(err.message)
    });
}

function deleteClassifier(classifier_id) {
    event.stopPropagation();
    axiosInstance.delete("/" + classifier_id)
     .then(function (res) {
       database.ref("/classifiers/" + classifier_id).remove();
       if ($("#classifier_list").html() == "")
         selected = {
           classifier_id: null,
           classifier_name: null
         }
     })
     .catch(function (err) {
       alert(err.message)
     });
}

$("#upload_dataset").click(uploadDataset);
$("#classifyText").click(processClassify);
retrieveClassifiers();