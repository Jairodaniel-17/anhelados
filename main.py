import os
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI

load_dotenv()
##################################################
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
BASE_URL = "https://api.openai.com/v1"
DB_PATH = "./anhelados.db"

# Configuración del modelo OpenAI
llm = ChatOpenAI(api_key=API_KEY, model=MODEL, base_url=BASE_URL, max_tokens=1024)


class ArithmeticOperations:
    @staticmethod
    def multiply(a: int, b: int) -> int:
        return a * b

    @staticmethod
    def add(a: int, b: int) -> int:
        return a + b

    @staticmethod
    def subtract(a: int, b: int) -> int:
        return a - b

    @staticmethod
    def divide(a: int, b: int) -> int:
        return a / b


class SQLExecutor:
    @staticmethod
    def query(sql_query: str):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                results = cursor.fetchall()
                if results:
                    return results
                else:
                    return "No existe registro"
        except sqlite3.Error as e:
            return f"Error en la consulta: {e}"


# Definición de esquemas de argumentos utilizando subclases de BaseModel
class MultiplyArgs(BaseModel):
    a: int = Field(..., description="The first integer.")
    b: int = Field(..., description="The second integer.")


class AddArgs(BaseModel):
    a: int = Field(..., description="The first integer.")
    b: int = Field(..., description="The second integer.")


class SubtractArgs(BaseModel):
    a: int = Field(..., description="The first integer.")
    b: int = Field(..., description="The second integer.")


class DivideArgs(BaseModel):
    a: int = Field(..., description="The first integer.")
    b: int = Field(..., description="The second integer.")


class SQLQueryArgs(BaseModel):
    sql_query: str = Field(..., description="The SQL query to execute.")


# Definición de herramientas utilizando StructuredTool
multiply_tool = StructuredTool.from_function(
    func=ArithmeticOperations.multiply,
    name="Multiply",
    description="Multiply two integers.",
    args_schema=MultiplyArgs,
)

add_tool = StructuredTool.from_function(
    func=ArithmeticOperations.add,
    name="Add",
    description="Add two integers.",
    args_schema=AddArgs,
)

subtract_tool = StructuredTool.from_function(
    func=ArithmeticOperations.subtract,
    name="Subtract",
    description="Subtract two integers.",
    args_schema=SubtractArgs,
)

divide_tool = StructuredTool.from_function(
    func=ArithmeticOperations.divide,
    name="Divide",
    description="Divide two integers.",
    args_schema=DivideArgs,
)


sql_tool = StructuredTool.from_function(
    func=SQLExecutor.query,
    name="SQLQuery",
    description="Execute an SQL query and return the results.",
    args_schema=SQLQueryArgs,
)

# Configuración del agente
tools = [multiply_tool, add_tool, subtract_tool, divide_tool, sql_tool]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente super amable te llamas Chispitas y asistes a la empresa Anhelados, siempre saludas, brindas información puntual y aqui te brindo datos para solucionar consultas del usuario, Ubicaciones en Lima, Perú: San Isidro, Miraflores, Surco y La Molina con horarios de Atencion de 9 AM a 10 PM. Ofrece helados artesanales, milkshakes, postres y bebidas. Promociones regulares y servicio de delivery disponible. Locales Específicos: San Isidro: Av. Javier Prado Este 1234, Tel: (01) 123-4567. Miraflores: Calle Alcanfores 567, Tel: (01) 234-5678. Surco: Av. Caminos del Inca 890, Tel: (01) 345-6789. La Molina: Av. La Molina 345, Tel: (01) 456-7890. Productos Destacados: Helados clásicos como vainilla, chocolate y fresa. Especialidades como lúcuma, maracuyá y chirimoya. Opciones veganas y gourmet. Promociones: 2x1 en helados clásicos los martes. Descuentos en helados gourmet al comprar 3 litros. Servicios Adicionales: Delivery y pedidos personalizados para eventos. Reservas para cumpleaños y reuniones. Siempre termina con un mensaje de agradecimiento por la preferencia.\n Además te brindo la estructura de la base de datos SQLITE en caso en la información antes mencionada no encuentres lo que pide el usuario: La base de datos tiene esta estructura SQLite: 1. almacen: ID_Almacen, ID_Insumo, Cantidad_Actual, Cantidad_Minima, Referencia: insumos (ID_Insumo), 2. clientes: ID_Proveedor, Nombre, Contacto, Direccion, 3. detalle_pedidos: ID_Pedido, ID_Insumo, Cantidad, Costo_Unitario, Referencias: pedidos_proveedor (ID_Pedido), insumos (ID_Insumo), 4. detalle_ventas: ID_Venta, ID_Producto, Cantidad, Precio_Unitario, Referencias: ventas (ID_Venta), productos (ID_Producto), 5. empleados: ID_Empleado, Nombre, Apellido, Sueldo, Fecha_Inicio, 6. gastos: ID_Gasto, Tipo, Monto, Fecha, 7. gastos_imprevistos: ID_Gasto_Imprevisto, Descripcion, Monto, Fecha,8. insumos: ID_Insumo, Descripcion, Tipo, Costo, Cantidad, 9. maquinas: ID_Maquina, Tipo, Capacidad, Consumo_Energetico, 10. pedidos_proveedor: ID_Pedido, ID_Proveedor, Fecha, Total, Referencia: clientes (ID_Proveedor), 11. produccion: ID_Produccion, ID_Maquina, Fecha, Cantidad_Producida, Referencia: maquinas (ID_Maquina), 12. productos: ID_Producto, Nombre, Precio, 13. registro_energetico: ID_Registro, ID_Maquina, Fecha, Consumo, Referencia: maquinas (ID_Maquina), 14. ventas: ID_Venta, Fecha, ID_Empleado, Total, Referencia: empleados (ID_Empleado). \n Además tienes que usar las herramientas cuando sea necesario. Tus respuestas deben ser claras y precisas.",
        ),
        ("user", "{question}"),
        ("assistant", "{agent_scratchpad}"),  # Incluir agent_scratchpad en el prompt
    ]
)

agent = create_openai_functions_agent(llm, tools, prompt)
chain = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)


def ask_agent(question: str) -> str:
    result = chain.invoke({"question": question})
    return result["output"]


st.title("Chatbot de Anhelados")

if "history" not in st.session_state:
    st.session_state["history"] = []

pregunta = st.chat_input("Escribe tu consulta...")

if pregunta:
    st.session_state["history"].append({"role": "user", "content": pregunta})
    respuesta = ask_agent(pregunta)
    st.session_state["history"].append({"role": "ai", "content": respuesta})

for message in st.session_state["history"]:
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="👩‍💻"):
            st.write(message["content"])
    else:
        with st.chat_message(name="ai", avatar="🍦"):
            st.write(message["content"])
