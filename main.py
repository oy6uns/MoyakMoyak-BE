import os
import numpy as np
import faiss
import openai
from dotenv import load_dotenv, find_dotenv

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/", StaticFiles(directory="public", html = True), name="static")

load_dotenv()
# 환경 변수에서 API 키 가져오기
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai.api_key = api_key

# 현재 디렉토리 설정
current_dir = os.getcwd()
vector_db_dir = os.path.join(current_dir, 'VectorDB')

# FAISS 인덱스 로드
index = faiss.read_index(os.path.join(vector_db_dir, 'vector_db.index'))
index_dimension = index.d  # 인덱스의 차원 확인

# 파일 경로 로드
with open(os.path.join(vector_db_dir, 'file_paths.txt'), 'r', encoding='utf-8') as f:
    file_paths = [line.strip() for line in f]

# 모든 문서 로드
documents = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as file:
        documents.append(file.read())

# 파일 경로와 링크 URL 매핑
file_path_to_url = {
    'Pill_Data/게보린정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A1270A0060',
    'Pill_Data/닥터베아제정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11AOOOOO0816',
    'Pill_Data/동성정로환에프정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2018080700007',
    'Pill_Data/돌코락에스장용정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2009092300055',
    'Pill_Data/로프민정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A1660A0057',
    'Pill_Data/머시론정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11ABBBBB2499',
    'Pill_Data/멜리안정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11AKP08G3641',
    'Pill_Data/모드콜에스연질캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2012050900002',
    'Pill_Data/뮤코팩트정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2009071505405',
    'Pill_Data/바스크롱캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2019072300058',
    'Pill_Data/벤포벨정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2015121700002',
    'Pill_Data/보나링츄어블정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016021200001',
    'Pill_Data/센시아정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2012011900004',
    'Pill_Data/스트렙실허니레몬트로키.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2009092800048',
    'Pill_Data/액티리버모닝연질캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2009090700436',
    'Pill_Data/액티피드정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A0500A0068',
    'Pill_Data/어린이용타이레놀정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A4380A0042',
    'Pill_Data/에세푸릴캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A1310A0170',
    'Pill_Data/우루사정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A0430A0174',
    'Pill_Data/우먼스타이레놀정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2022021100001',
    'Pill_Data/이가탄에프캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=x1yz9ddq5chfc',
    'Pill_Data/이지엔6이브연질캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2013011800015',
    'Pill_Data/인사돌플러스정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11AKP08F0848',
    'Pill_Data/인후신캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=schyhyjdytlpp',
    'Pill_Data/임팩타민정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=trna7vrzacd28',
    'Pill_Data/지르텍정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11ABBBBB2527',
    'Pill_Data/카베진코와알파정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2018100500006',
    'Pill_Data/타세놀이알서방정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11ALLLLL0032',
    'Pill_Data/타이레놀정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2021082400002',
    'Pill_Data/타이레놀콜드-에스정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2021101800010',
    'Pill_Data/판시딜캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2010101100004',
    'Pill_Data/판피린티정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A0150A0650',
    'Pill_Data/화이투밴큐노즈연질캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2013010400010',
    'Pill_Data/화이투벤씨플러스캡슐.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2013010700008',
    'Pill_Data/훼스탈플러스정.txt': 'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=A11A0740B0009',
}

# 임베딩 생성 함수
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)


def search(query, top_k=3):
    # 쿼리 벡터화
    query_embedding = get_embedding(query).reshape(1, -1)

    # 차원 일치 여부 확인
    if query_embedding.shape[1] != index_dimension:
        raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {index_dimension}")

    # FAISS 인덱스에서 유사한 문서 검색
    D, I = index.search(query_embedding, top_k)

    results = [(file_paths[i], documents[i]) for i in I[0]]
    return results


def generate_answer(query, top_k=2):
    # 관련 문서 검색
    relevant_docs = search(query, top_k)

    # 관련 문서들을 하나의 문자열로 결합
    context = "\n\n".join(doc for _, doc in relevant_docs)
    references = []
    for path, _ in relevant_docs:
        path = path.split('/')[-2] + '/' + path.split('/')[-1]
        print(path)
        if path in file_path_to_url:
            references.append(file_path_to_url[path])
    references = '\n'.join(references)

    system_message = ("이 챗봇은 알약에 대해 답변하기 위해 만들어진 챗봇입니다."
                          "챗봇은 데이터베이스에 있는 내용들을 활용해서 답변을 하면됩니다."
                          "데이터베이스에는 시중에서 판매되는 상비약들에 대한 정보가 담겨있어."
                          "상비약들의 정보를 확인해서 사용자의 질문에 대답해주면 돼."
                          "최대한 간략하고 핵심정보만을 전달해주면 좋을 것 같고, 만약에 추가 정보가 필요하다면 사용자에게 물어보면 돼.")


    # OpenAI GPT-4 모델을 사용하여 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Here are some documents:\n\n{context}\n\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=1000  # max_tokens 값을 증가시켜 잘리는 문제 방지
    )

    answer = response.choices[0].message['content'].strip()
    return answer, references

app = FastAPI()

@app.post("/final", status_code=201)
async def makeIcon_URL(query: str):
    # Return the URL
    answer, references = generate_answer(query)
    return {"answer": answer, "references": references}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)
