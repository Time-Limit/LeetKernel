#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_41_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<41> mm_instantiator;

public:
  UnalignedM_41_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_41_MMInstantiatorWrapper__;

}  // namespace LLMMM
