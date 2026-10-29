# 원본 프로젝트와 포함 코드

이 레포는 아래 프로젝트를 바탕으로 현재 AM-ARM200 + ORCA Hand v1 장치용 수정본과 수집 UI를 함께 제공합니다. 클론만으로 모델·수정된 하드웨어 코드·현재 보정값을 받을 수 있도록 관련 파일을 포함했습니다.

- **ORCA Core**: <https://github.com/orcahand/orca_core> — `orca_core/`. 원본 MIT 라이선스는 [orca_core/LICENSE](orca_core/LICENSE)에 보관돼 있습니다.
- **ORCA Hand Description**: <https://github.com/orcahand/orcahand_description> — `orcahand_description/`. 원본 MIT 라이선스는 [orcahand_description/LICENSE](orcahand_description/LICENSE)에 보관돼 있습니다.
- **ORCA Teleop**: <https://github.com/orcahand/orca_teleop> — `orca_teleop/`. 기존 코드·README·프로젝트 메타데이터를 유지하며 로컬 수정 사항을 포함합니다. 해당 원본 사본에는 별도 LICENSE 파일이 없어 다른 프로젝트의 MIT 라이선스를 이 코드 전체에 자동 적용하지 않습니다.
- **AM-ARM**: <https://github.com/liyiteng/AM-ARM> — AM 모델에서 가져온 팔 구성·메시는 `orca_am200/`에 포함돼 있습니다. 원본 Apache-2.0 라이선스를 [licenses/AM-ARM-Apache-2.0.txt](licenses/AM-ARM-Apache-2.0.txt)에 보관했습니다.

수집 UI, 로컬 녹화 제어, 카메라 역할 지정, 현재 로봇의 모터 매핑/캘리브레이션, 연속 핀치 보정과 팔 명령 제한 등은 이 작업 공간에서 추가·수정한 내용입니다. 포함된 각 원본 라이선스·저작권 표시는 해당 파일에 적용됩니다.
