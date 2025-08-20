# LoRA Adapter Loader: TODO 구현 가이드

이 문서는 `LoRAModelManager` 클래스 내 `_load_weights` 메서드를 완성하기 위해 구현해야 할 `TODO` 항목들을 설명합니다. 목표는 `adapter_model.safetensors` 파일에서 LoRA 가중치를 올바르게 로드하고, 스케일링을 적용한 후, 미리 정의된 `self.lora_weights` 딕셔너리에 저장하는 것입니다.

## 📌 주요 구현 목표

`_load_weights` 메서드 내 `for` 루프에서 `safetensors` 파일의 각 키(key)를 분석하여 LoRA 가중치를 처리해야 합니다. 키의 구조는 일반적으로 다음과 같습니다.

`base_model.model.layers.{layer_idx}.{block_type}.{op_name}.{adapter_type}.weight`

이 구조를 바탕으로 아래의 `TODO` 항목들을 구현해야 합니다.

---

## ✅ `TODO` 구현 목록

### 1. Transformer Layer 가중치 처리

`if parts[3] == 'layers':` 블록 내의 `TODO` 항목들을 구현해야 합니다.

#### 📝 **1-1: 키(key) 정보 추출**

`safetensors` 키를 파싱하여 레이어 인덱스, 블록 타입, 연산 이름, 어댑터 타입을 추출해야 합니다.

* **구현 대상:**
    ```python
    # TODO: Extract layer index, block type, op name, and adapter_type from parts
    layer_idx = ...
    block_type = ...
    op_name = ...
    adapter_type = ...
    ```

#### 📝 **1-2: `lora_B` 가중치에 스케일링 적용**

추출된 `adapter_type`이 `lora_B`인 경우, `_read_scaling` 메서드에서 읽어온 `scale` 값을 텐서에 곱해야 합니다.

* **구현 대상:**
    ```python
    # TODO: Apply scaling factor to t if this is a lora_B
    ...
    ```

#### 📝 **1-3: `self.lora_weights`에 텐서 저장**

추출된 인덱스 정보(`layer_idx`, `block_idx`, `op_idx`, `adapter_idx`)를 사용하여 처리된 텐서를 `self.lora_weights['layers']`의 올바른 위치에 저장합니다.

* **구현 대상:**
    ```python
    # TODO: Store tensor in self.lora_weights
    ...
    ```

---

### 2. LM Head 가중치 처리

`elif parts[2] == 'lm_head':` 블록 내의 `TODO` 항목들을 구현해야 합니다.

#### 📝 **2-1: `lora_B` 가중치에 스케일링 적용**

LM Head의 어댑터 타입이 `lora_B`인 경우, `scale` 값을 텐서에 곱합니다.

* **구현 대상:**
    ```python
    # TODO: Apply scaling factor to t if this is a lora_B
    ...
    ```

#### 📝 **2-2: 어댑터 인덱스 결정**

어댑터 타입(`lora_A` 또는 `lora_B`)에 따라 저장할 인덱스(0 또는 1)를 결정합니다.

* **구현 대상:**
    ```python
    # TODO: Determine an index based on adapter_type
    ...
    ```

#### 📝 **2-3: `self.lora_weights`에 텐서 저장**

결정된 인덱스를 사용하여 `self.lora_weights['lm_head']`에 텐서를 저장합니다.

* **구현 대상:**
    ```python
    # TODO: Store tensor in self.lora_weights
    ...
    ```
