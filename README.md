​AURA (Autonomous Urban Risk Analyzer)
=============

## 📖 목차

* 1.[프로젝트 개요](#프로젝트-개요)
* 2.[시스템 아키텍쳐](#시스템-아키텍쳐)
* 3.[AI 모델 개발](#AI-모델-개발)
* 4.[성능 평가 및 결과](#성능-평가-및-결과)
* 5.[결론](#결론)
* 6.[향후 계획](#향후-계획)

***

## 1.프로젝트 개요

<p align="center">
  <img width="351" height="379" src="https://github.com/user-attachments/assets/f1b22e6b-c014-4598-a939-6db46ee2cf55" />
</p>

### 시스템 목적
AI 기반 이벤트 식별 시스템을 개발하여, 화재, 폭행, 실종자 확인, 흡연자, 무단투기자 등 다양한 이벤트를 실시간으로 감지하고 식별합니다.

*"본 보고서는 AI 기반 이벤트 식별 시스템을 개발한 결과에 대해 상세히 다룹니다. 보고서는 시스템 개발 과정, 성능 평가 결과, 그리고 향후 계획을 포함하여 프로젝트의 전반적인 내용을 제시합니다."*

### 개발 목표
AI 기반 시스템이 특정 환경(주간, 야간, 다양한 기상 조건)에서 주요 순찰 이벤트를 식별하는 것
<p align="center">
  <img width="1038" height="331" src="https://github.com/user-attachments/assets/9dbeff63-9ee8-4c4c-9853-a937750876d7" />
</p>

***

## 2.시스템 아키텍쳐
<details>
  <summary><span style="color:blue; text-decoration:underline; cursor:pointer;">HW_Architecture</span></summary>
  <p align="center">
    <img width="600" src="https://github.com/user-attachments/assets/ae5db6d8-959a-46bc-8638-1745565d26ca" />
  </p>
</details>

<details>
  <summary><span style="color:blue; text-decoration:underline; cursor:pointer;">SW_Architecture</span></summary>
  <p align="center">
    <img width="600" src="https://github.com/user-attachments/assets/a89ce0e3-d57c-4bf3-b77c-8b2300b57b5f" />
  </p>
</details>

<img width="991" height="563" alt="image" src="https://github.com/user-attachments/assets/3b7c6519-7e44-457f-9d2c-0e1915c06530" />

## 3.GUI 구현



***

## 4.AI 모델 개발

### 1.화재 감지
YOLOv8 Nano (640x640, 3 epoch, bachsize 16)
### 2.폭행 감지

### 3.실종자 확인

### 4.흡연자 감지
YOLOv8 Nano, (640x640, 70 epochs, batch 16)

### 5.무단투기자 식별
* 영상내의 무단투기 행위에 대한 정확성을 수치로 인식합니다 (0~1)
<p align="center">
  <img width="600" src="  https://drive.google.com/file/d/1hZldRzCdQJ4kxARSFdek8sjoseQrg72S/view?usp=drive_link" />
</p>
* I3D (Slow_r50) (224x224, 17 epochs, batch size 2, clip length 8)

### 6.실신자 식별

***


## 5.성능 평가 및 결과


