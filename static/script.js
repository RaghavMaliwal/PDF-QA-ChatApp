document.getElementById("uploadBtn").addEventListener("click", async () => {
  const input = document.getElementById("pdfInput");
  const fileList = input.files;

  if (fileList.length === 0) {
    alert("Please select at least one PDF.");
    return;
  }

  const formData = new FormData();
  for (let file of fileList) {
    formData.append("files", file);
  }

  // Show progress in chat
  const chatBox = document.getElementById("chatBox");
  chatBox.innerHTML += `<div class="text-center text-gray-600">Uploading files...</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch("http://localhost:8000/upload/", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();

    // Show feedback in chat after upload is complete
    chatBox.innerHTML += `<div class="text-center text-green-600">Uploaded: ${result.uploaded.join(", ")}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    console.log(`Files uploaded successfully: ${result.uploaded.join(", ")}`);
  } catch (error) {
    // Show error in chat if upload fails
    chatBox.innerHTML += `<div class="text-center text-red-600">Error uploading files. Please try again.</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    console.error("File upload failed:", error);
  }
});

document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("userInput");
  const chatBox = document.getElementById("chatBox");

  const userText = input.value;
  chatBox.innerHTML += `<div class="text-right text-blue-800"><strong>You:</strong> ${userText}</div>`;
  input.value = "";

  // Show processing message in chat
  chatBox.innerHTML += `<div class="text-center text-gray-600">Processing your question...</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch(
      `http://localhost:8000/query/`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userText })
      }
    );
    const data = await res.json();

    // Remove "Processing..." message before showing the response
    document.querySelectorAll(".text-gray-600").forEach(el => el.remove());

    // Display the AI answer
    chatBox.innerHTML += `<div class="text-left text-green-700"><strong>AI:</strong> ${data.response}</div>`;

    // Display the sources of the answer
    const sourcesDiv = document.createElement("div");
    sourcesDiv.classList.add("text-left", "text-gray-600");
    sourcesDiv.innerHTML = "<strong>Sources:</strong>";

    const sourcesList = document.createElement("ul");
    data.sources.forEach(source => {
      const listItem = document.createElement("li");
      listItem.innerText = `Document: ${source.document_name}, Chunk ID: ${source.chunk_id}`;
      sourcesList.appendChild(listItem);
    });

    sourcesDiv.appendChild(sourcesList);
    chatBox.appendChild(sourcesDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    console.log(`Question: ${userText}`);
    console.log(`AI Response: ${data.response}`);
    console.log(`Sources: ${JSON.stringify(data.sources)}`);
  } catch (error) {
    chatBox.innerHTML += `<div class="text-center text-red-600">Error processing the question. Please try again.</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    console.error("Question processing failed:", error);
  }
});
