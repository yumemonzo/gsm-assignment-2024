document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault(); // 기본 폼 제출 동작 방지
  
    const fileInput = document.getElementById("image-upload");
    const file = fileInput.files[0]; // 사용자가 선택한 파일 가져오기
  
    if (!file) {
      alert("Please upload an image!"); // 파일이 없는 경우 경고
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file); // FormData에 파일 추가
  
    try {
      // FastAPI의 /predict/ 엔드포인트로 POST 요청
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST", // HTTP POST 메서드 사용
        body: formData, // FormData 전송
      });
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`); // 서버 에러 처리
      }
  
      const result = await response.json(); // 응답을 JSON으로 파싱
      displayResult(result, URL.createObjectURL(file)); // 결과와 업로드된 이미지 표시
    } catch (error) {
      alert(`Failed to get prediction: ${error.message}`); // 에러 발생 시 경고
    }
  });
  
  // 예측 결과와 업로드된 이미지를 화면에 표시
  function displayResult(result, imageUrl) {
    const resultDiv = document.getElementById("result"); // 결과를 표시할 div 요소
    const imageElement = document.getElementById("uploaded-image"); // 업로드된 이미지 요소
  
    // 예측 결과 HTML로 표시
    resultDiv.innerHTML = `
      <h2>Prediction Result</h2>
      <p><strong>Class:</strong> ${result.class}</p> <!-- 예측된 클래스 -->
      <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p> <!-- 신뢰도 -->
    `;
    imageElement.src = imageUrl; // 업로드된 이미지의 URL 설정
    imageElement.hidden = false; // 이미지 표시
  }
  