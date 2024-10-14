from flask import Flask, request, jsonify,abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction, TemplateSendMessage,CarouselTemplate,CarouselColumn,URIAction,ButtonsTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j import GraphDatabase, basic_auth
import json
from flask_ngrok import run_with_ngrok

# ตั้งค่าการเชื่อมต่อ Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "1"))

# ใช้โมเดล SentenceTransformer สำหรับการคำนวณความเหมือนของข้อความ
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# ฟังก์ชันเพื่อบันทึกประวัติการแชทใน Neo4j
def save_chat_history(user_id, message, reply):
    cypher_query = '''
    MERGE (u:User {userId: $user_id})
    CREATE (um:UserMessage {message: $message, timestamp: timestamp()})
    CREATE (br:BotReply {reply: $reply, timestamp: timestamp()})
    CREATE (u)-[:SENT]->(um)
    CREATE (u)-[:RECEIVED]->(br)
    CREATE (um)-[:TRIGGERED]->(br)
    '''
    with driver.session() as session:
        session.run(cypher_query, user_id=user_id, message=message, reply=reply)

# ฟังก์ชันคำนวณความเหมือนของข้อความ
def compute_similar(corpus, sentence):
    a_vec = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    b_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    similarities = np.inner(b_vec.cpu().numpy(), a_vec.cpu().numpy())
    return similarities

# ฟังก์ชัน Quick Reply Menu
def quick_reply_menu(tk):
    print("Building quick reply menu...")  # Debug message
    quick_reply_buttons = [
        QuickReplyButton(action=MessageAction(label="แนะนำตัว", text="แนะนำตัว")),
        QuickReplyButton(action=MessageAction(label="รุ่นรถ", text="รุ่นรถ")),
        QuickReplyButton(action=MessageAction(label="ราคารถ ", text="ราคารถ")),
        QuickReplyButton(action=MessageAction(label="เว็บไซต์ ", text="เว็บไซต์")),
    ]
    quick_reply = QuickReply(items=quick_reply_buttons)
    line_bot_api.reply_message(tk, TextSendMessage(text="หากต้องการสอบถามสามารถเลือกคำถามจากข้างล่างข้อความนี้ได้เลยค่ะ", quick_reply=quick_reply))
    print("Quick reply sent successfully.")

def save_user_name(user_id, name):
    print(f"Saving user name: {name} for userId: {user_id}")  # Debug message
    cypher_query = '''
    MERGE (u:User {userId: $user_id})
    ON CREATE SET u.name = $name
    ON MATCH SET u.name = $name
    '''
    with driver.session() as session:
        session.run(cypher_query, user_id=user_id, name=name)
        print("User name saved successfully.")  # Debug message


def get_user_name(user_id):
    cypher_query = '''
    MATCH (u:User {userId: $user_id})
    RETURN u.name AS name
    '''
    with driver.session() as session:
        result = session.run(cypher_query, user_id=user_id)
        single_result = result.single()
        if single_result:
            return single_result['name']
        else:
            return None

def check_hrv_models(sentence, car_data):
    # แยกข้อความที่ผู้ใช้พิมพ์ด้วยช่องว่าง (space)
    query_parts = sentence.split(" ", 3)  # แยกข้อความเป็น 4 ส่วนโดยใช้ช่องว่าง
    print(f"Query Parts: {query_parts}")  # แสดงผลการแยกประโยค

    # ตรวจสอบความยาวของ query_parts ก่อนเข้าถึง
    if len(query_parts) >= 1:  # ต้องมีอย่างน้อย 1 ส่วน
        car_name_query = query_parts[0]  # ส่วนแรกใช้ตรวจสอบชื่อรถ
        grade_name_query = " ".join(query_parts[1:]) if len(query_parts) > 1 else None  # ตรวจสอบเกรดถ้ามี

        for car in car_data:
            if car_name_query.lower() in car['car_name'].lower():  # ตรวจสอบว่าชื่อรถตรงกันหรือไม่ (ไม่สนใจตัวพิมพ์)
                matched_grades = []
                for grade in car.get('grades', []):
                    # ถ้ามีเกรด ถ้ามีการระบุเกรดให้ตรวจสอบ
                    if grade_name_query and grade_name_query in grade['grade_name']:
                        matched_grades.append(grade)  # เก็บเกรดที่ตรงกัน

                # หากมีเกรดที่ตรงกันให้แสดงผล
                if matched_grades:
                    # เลือกเกรดที่มีราคาต่ำสุดหรือตัวแรกในรายการ matched_grades
                    model_name = matched_grades[0].get('grade_name')
                    price = matched_grades[0].get('grade_price')
                    print(f"Matched Model: {model_name}, Price: {price}")  # แสดงผลการจับคู่
                    if model_name:
                        price_text = f"ราคา: {price}" if price not in ['', 'N/A', 'ไม่แสดงราคา'] else f"ไม่แสดงราคา สามารถดูข้อมูลเพิ่มเติมได้ที่ลิ้ง: {car['model_link']}"
                        return f"{car['car_name']}\nโมเดล {model_name} {price_text}"

                # ถ้าไม่มีเกรดตรงกันและไม่ได้ระบุเกรด ให้แสดงราคาเกรดแรก
                if not grade_name_query and car.get('grades'):
                    first_grade = car['grades'][0]
                    model_name = first_grade.get('grade_name')
                    price = first_grade.get('grade_price')
                    
                    # ถ้าราคาเป็น 'ไม่แสดงราคา' จะเพิ่มข้อความพร้อมลิ้ง
                    price_text = f"ราคา: {price}" if price not in ['', 'N/A', 'ไม่แสดงราคา'] else f"ไม่แสดงราคา สามารถดูข้อมูลเพิ่มเติมได้ที่ลิ้ง: {car['model_link']}"
                    
                    return f"{car['car_name']}\nโมเดล {model_name} {price_text}"

        return f"{car['car_name']} ไม่มีโมเดลที่แนะนำ"
    else:
        return "กรุณาพิมพ์ข้อมูลในรูปแบบที่ถูกต้อง เช่น 'HR-V HEV EL' หรือ 'HR-V E'"


def compute_response(sentence, car_data, tk=None, user_id=None):
    
    responses = {
        "สวัสดี": "สวัสดีค่ะ ต้องการสอบถามเกี่ยวกับรถ Honda เรื่องอะไรคะ หากต้องการคำแนะนำในการถามให้พิมพ์คำว่า 'สอบถาม' ",
        "ลาก่อน": "หากต้องการทราบอะไรเพิ่มเติม สามารถสอบถามข้อมูลเพิ่มเติมได้ค่ะ",
        "ขอบคุณ": "ยินดีค่ะ",
        
        "แนะนำตัว": "ฉันชื่อเวล เป็นบอทแนะนำรถ Honda หากคุณต้องการแจ้งชื่อตัวคุณเอง สามารถพิมพ์ว่า 'ฉันชื่อ(ตามด้วยชื่อ)' หรือ 'ผมชื่อ(ตามด้วยชื่อ)' และหากต้องการถามชื่อของตัวเอง สามารถพิมพ์ว่า 'ชื่ออะไร' หรือ 'ชื่อฉันคือ' หรือ 'ชื่อผมคือ' ได้เลยค่ะ",
        "เว็บไซต์": "คุณสามารถเยี่ยมชมเว็บไซต์ Honda ได้ที่: https://www.honda.co.th/",
        "มีรุ่นอื่นอีกไหม": "สามารถดูรายละเอียดรถรุ่นอื่นๆเพิ่มเติมของ Honda ได้ที่ : https://www.honda.co.th/",
    }

    # ข้อความที่ต้องการตรวจสอบสำหรับการแนะนำรถยนต์
    car_queries = {
        "มีรถรุ่นอะไรบ้าง": "มีรถรุ่นอะไรบ้าง",
        "มีรถอะไรบ้าง": "มีรถอะไรบ้าง",
        "มีรถรุ่นอะไรบ้าง": "รุ่นรถ",
        "ราคารถ": "ราคารถ",
        "ราคา":"ราคา",
        "ราคารถ": "ราคารถแต่ะละรุ่น",

        "Honda HR-V": "Honda HR-V",
        "โมเดลHR-V E": "โมเดลHR-V E",
        "โมเดลHR-V EL": "โมเดลHR-V EL", 
        "โมเดลHR-V RS": "โมเดลHR-V RS",

        "Honda Civic": "Honda Civic",
        "Civic": "Civic",
        "โมเดลCivic EL+": "โมเดลCivic EL+",
        "โมเดลCivic HEV EL+": "โมเดลCivic HEV EL+",
        "โมเดลCivic RS": "โมเดลCivic RS",

        "Honda N1": "Honda N1",
        "N1": "N1",
        "โมเดลN1": "โมเดลN1",
        
        "Honda City": "Honda City",
        "City": "City",
        "โมเดลCity S+":"โมเดลCity S+", 
        "โมเดลCity SV":"โมเดลCity SV", 
        "โมเดลCity RS":"โมเดลCity RS" , 
        "โมเดลCity HEV SV":"โมเดลCity HEV SV", 
        "โมเดลCity HEV RS":"โมเดลCity HEV RS" ,

        "Honda Accord": "Honda Accord",
        "Accord": "Accord",
        "โมเดลAccoed E":"โมเดลAccoed E" , 
        "โมเดลAccoed HEV EL":"โมเดลAccoed HEV EL", 
        "โมเดลAccoed HEV RS":"โมเดลAccoed HEV RS" ,

    }

    bot_phrases = ["ชื่อของเธอ", "ชื่อคุณคือ", "คุณชื่ออะไร", "เธอชื่ออะไร"]
    # เช็คว่าผู้ใช้ได้บอกชื่อแล้วหรือยัง
    for phrase in bot_phrases:
        if phrase in sentence:
                return f"ฉันชื่อเวลค่ะ หากต้องการคำแนะนำในการถามให้พิมพ์คำว่า 'สอบถาม'"

    

    name_phrases = ["ฉันชื่อ", "ผมชื่อ"]
    # เช็คว่าผู้ใช้ได้บอกชื่อแล้วหรือยัง
    for phrase in name_phrases:
        if phrase in sentence:
                name = sentence.replace(phrase, "").strip()  # ลบวลีที่ตรงกันออกและลบช่องว่าง
                save_user_name(user_id, name)  # บันทึกชื่อผู้ใช้
                return f"ยินดีที่ได้รู้จักค่ะ{name} ฉันชื่อเวลค่ะ หากต้องการคำแนะนำในการถามให้พิมพ์คำว่า 'สอบถาม'"
    
    ask_phrases = ["ชื่ออะไร", "ชื่อฉันคือ", "ชื่อผมคือ"]
    # เช็คว่าผู้ใช้ถามว่าชื่ออะไร
    for phrase in ask_phrases:
        if phrase in sentence:
            try:
                name = get_user_name(user_id)  # Retrieve the user's name from the database
                if name:
                    return f"ชื่อของคุณคือ{name}ค่ะ หากต้องการสอบถามเพิ่มเติมสามารถพิมพ์คำถามได้เลย"
                else:
                    return "คุณยังไม่ได้บอกชื่อค่ะ หากต้องการบอกชื่อให้พิมพ์ว่า 'ฉันชื่อ...'"
            except Exception as e:
                return "เกิดข้อผิดพลาดในการดึงชื่อของคุณ กรุณาลองอีกครั้ง"
    
    # คำนวณความคล้ายคลึงสำหรับคำถามที่เกี่ยวข้องกับรถยนต์
    similarities = {}
    for query, query_text in car_queries.items():
        similarity = compute_similar([query_text], sentence)[0]
        similarities[query] = similarity

    # หาคำถามที่มีความคล้ายคลึงมากที่สุด
    best_query = max(similarities, key=similarities.get)

    # ถ้าความคล้ายคลึงมากกว่า 0.6 จะถือว่าเป็นคำถามที่ตรงหรือใกล้เคียง
    if similarities[best_query] > 0.6:
        if best_query in ["มีรถอะไรบ้าง", "มีรถรุ่นอะไรบ้าง"] :
            car_names = [f"{i + 1}. {car['car_name']} \n({car.get('price', 'ไม่ทราบ') if car.get('price') not in ['', 'N/A'] else 'ไม่แสดงราคา'})" for i, car in enumerate(car_data)]
            return "รุ่นที่แนะนำ(รุ่นใหม่) :\n" + "\n".join(car_names) +"\nหากต้องการดูรุ่นอื่นๆเพิ่มเติมสามารถดูได้ที่ : https://www.honda.co.th/"

        if best_query in ["ราคารถ", "ราคา"]:
            car_names = [f"{i + 1}. {car['car_name']} " for i, car in enumerate(car_data)]
            return "สนใจสอบถามราคารุ่นไหนคะ :\n" + "\n".join(car_names)
        
        
        if best_query in ["Honda HR-V","HR-V", "รายละเอียดเกี่ยวกับ Honda HR-V", "รายละเอียดของHR-V" ]:
            hrv_models = []
            for car in car_data:
                if "Honda HR-V" in car['car_name']:
                    for i, grade in enumerate(car.get('grades', [])):
                        model_name = grade.get('grade_name')
                        if model_name:
                            hrv_models.append(f"{i + 1}. {model_name} ")
            return "Honda HR-V กรุณาเลือกโมเดลที่ต้องการ:\n" + "\n".join(hrv_models)+"\n*พิมพ์ในรูปแบบ: โมเดลHR-V HEV E" if hrv_models else "Honda HR-V ไม่มีโมเดลที่แนะนำ"
        

        if best_query in ["Honda Civic", "Civic","รายละเอียดเกี่ยวกับ Honda Civic", "รายละเอียดของCivic"]:
            civic_models = []
            for car in car_data:
                if "Honda Civic" in car['car_name']:
                    for i, grade in enumerate(car.get('grades', [])):
                        model_name = grade.get('grade_name')
                        price = grade.get('grade_price')
                        if model_name:
                            civic_models.append(f"{i + 1}. {model_name}")
            return "Honda Civic กรุณาเลือกโมเดลที่ต้องการ:\n" + "\n".join(civic_models) +"\n*พิมพ์ในรูปแบบ: โมเดลCivic HEV EL+"if civic_models else "Honda Civic ไม่มีโมเดลที่แนะนำ"
        
        if best_query in ["Honda N1", "N1","รายละเอียดเกี่ยวกับ Honda N1", "รายละเอียดของN1"]:
            n1_models = []
            for car in car_data:
                if "Honda e:N1" in car['car_name']:
                    for i, grade in enumerate(car.get('grades', [])):
                        model_name = grade.get('grade_name')
                        price = grade.get('grade_price')
                        if model_name:
                            n1_models.append(f"{i + 1}. {model_name}")
            return "Honda N1 กรุณาเลือกโมเดลที่ต้องการ:\n" + "\n".join(n1_models) +"\n*พิมพ์ในรูปแบบ: โมเดลN1"if n1_models else "Honda N1 ไม่มีโมเดลที่แนะนำ"
        
        if best_query in ["Honda City", "City","รายละเอียดเกี่ยวกับ Honda City", "รายละเอียดของCity"]:
            city_models = []
            for car in car_data:
                if "Honda City Hatchback" in car['car_name']:
                    for i, grade in enumerate(car.get('grades', [])):
                        model_name = grade.get('grade_name')
                        price = grade.get('grade_price')
                        if model_name:
                            city_models.append(f"{i + 1}. {model_name}")
            return "Honda City กรุณาเลือกโมเดลที่ต้องการ:\n" + "\n".join(city_models) +"\n*พิมพ์ในรูปแบบ: โมเดลCity HEV SV"if city_models else "Honda City ไม่มีโมเดลที่แนะนำ"
        
        if best_query in ["Honda Accord", "Accord","รายละเอียดเกี่ยวกับ Honda Accord", "รายละเอียดของAccord"]:
            accord_models = []
            for car in car_data:
                if "Honda Accord" in car['car_name']:
                    for i, grade in enumerate(car.get('grades', [])):
                        model_name = grade.get('grade_name')
                        price = grade.get('grade_price')
                        if model_name:
                            accord_models.append(f"{i + 1}. {model_name}")
            return "Honda Accord กรุณาเลือกโมเดลที่ต้องการ:\n" + "\n".join(accord_models) +"\n*พิมพ์ในรูปแบบ: โมเดลAccord HEV E"if accord_models else "Honda Accord ไม่มีโมเดลที่แนะนำ"



        if sentence.strip().startswith("โมเดล"):
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                result = check_hrv_models(trimmed_sentence, car_data)
                if result:
                    return result
                
        if best_query in ["โมเดลHR-V E", "โมเดลHR-V EL", "โมเดลHR-V RS"]:
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                # ตัดเครื่องหมาย + ออกจากคำค้นหาก่อนเรียกฟังก์ชัน
                processed_sentence = trimmed_sentence.replace("+", "").strip()  # ตัดเครื่องหมาย + และลบช่องว่างข้างหน้า
                result = check_hrv_models(processed_sentence, car_data)
                
                if result:
                    return result

        if best_query in ["โมเดลCivic EL+", "โมเดลCivic HEV EL+", "โมเดลCivic RS"]:
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                # ตัดเครื่องหมาย + ออกจากคำค้นหาก่อนเรียกฟังก์ชัน
                processed_sentence = trimmed_sentence.replace("+", "").strip()  # ตัดเครื่องหมาย + และลบช่องว่างข้างหน้า
                result = check_hrv_models(processed_sentence, car_data)
                
                if result:
                    return result
                
        if best_query in ["โมเดลN1"]:
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                # ตัดเครื่องหมาย + ออกจากคำค้นหาก่อนเรียกฟังก์ชัน
                processed_sentence = trimmed_sentence.replace("+", "").strip()  # ตัดเครื่องหมาย + และลบช่องว่างข้างหน้า
                result = check_hrv_models(processed_sentence, car_data)
                
                if result:
                    return result

        if best_query in ["โมเดลCity S+", "โมเดลCity SV", "โมเดลCity RS" , "โมเดลCity HEV SV", "โมเดลCity HEV RS" ]:
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                # ตัดเครื่องหมาย + ออกจากคำค้นหาก่อนเรียกฟังก์ชัน
                processed_sentence = trimmed_sentence.replace("+", "").strip()  # ตัดเครื่องหมาย + และลบช่องว่างข้างหน้า
                result = check_hrv_models(processed_sentence, car_data)
                
                if result:
                    return result     
                
        if best_query in ["โมเดลAccord HEV E", "โมเดลAccord HEV EL", "โมเดลAccord HEV RS" ]:
            trimmed_sentence = sentence[len("โมเดล"):].strip()  # ตัดคำว่า "โมเดล" และลบช่องว่างข้างหน้า
            print(f"Trimmed Sentence: '{trimmed_sentence}'")  # แสดงข้อความที่ตัดแล้ว
            
            if trimmed_sentence:  # ตรวจสอบว่า trimmed_sentence ไม่เป็นสตริงว่าง
                # ตัดเครื่องหมาย + ออกจากคำค้นหาก่อนเรียกฟังก์ชัน
                processed_sentence = trimmed_sentence.replace("+", "").strip()  # ตัดเครื่องหมาย + และลบช่องว่างข้างหน้า
                result = check_hrv_models(processed_sentence, car_data)
                
                if result:
                    return result 

    # คำนวณความคล้ายคลึงสำหรับคำตอบทั่วไป
    similarities = {}
    for corpus, response_msg in responses.items():
        similarity = compute_similar([corpus], sentence)[0]
        similarities[corpus] = similarity
        print(f"Comparing '{sentence}' to '{corpus}': Similarity = {similarity}")

    best_corpus = max(similarities, key=similarities.get)
    if similarities[best_corpus] > 0.6:
        return responses[best_corpus]
    
    if sentence == "สอบถาม" and tk:
        print(f"Quick reply triggered for message: {sentence}, reply token: {tk}")
        quick_reply_menu(tk)
        return None  # Since reply is handled by quick_reply_menu

    # General responses
    if sentence in responses:
        return responses[sentence]

    return "ขอโทษค่ะ ฉันไม่เข้าใจข้อความของคุณ หากต้องการคำแนะนำในการถามให้พิมพ์คำว่า 'สอบถาม'"


# อ่านข้อมูลจากไฟล์ car_data.json
def load_car_data(filename='car_data.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# ตั้งค่า LINE API
access_token = ''
secret = ''
line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret)


app = Flask(__name__)

# กำหนดเส้นทางสำหรับ webhook ของ LINE
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    signature = request.headers['X-Line-Signature']

    try:
        handler.handle(body, signature)

        json_data = json.loads(body)
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId']

        # โหลดข้อมูลรถยนต์
        car_data = load_car_data()

        # สร้างการตอบกลับ
        response_msg = compute_response(msg, car_data, tk=tk, user_id=user_id)


        if response_msg:  # ถ้าไม่มี Quick Reply จะแสดงข้อความปกติ
            # บันทึกประวัติการแชทลงใน Neo4j
            save_chat_history(user_id, msg, response_msg)

            # ตอบกลับผู้ใช้
            line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))

        print(f"User message: {msg}, Bot response: {response_msg}, UserId: {user_id}")

    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print(f"Error: {e}")

    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
